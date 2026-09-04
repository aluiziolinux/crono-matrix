"""Aplicacao FastAPI e rotas HTMX do Crono launcher."""

from pathlib import Path
import asyncio
import html
import ipaddress
import json
import urllib.parse
import os
import signal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from launch_model_core import _hf_format_bytes, _hf_format_params
from web.services import LauncherWebState, EvalRunner


ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.filters["bytes"] = _hf_format_bytes
templates.env.filters["params"] = _hf_format_params

state = LauncherWebState()
eval_runner = EvalRunner()
app = FastAPI(title="Crono Matrix Launcher", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.on_event("shutdown")
def shutdown_processes():
    state.stop_server()
    eval_runner.stop_run()


async def form_values(request: Request):
    body = (await request.body()).decode("utf-8", errors="replace")
    parsed = urllib.parse.parse_qs(body, keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def context(request: Request, **values):
    return {"request": request, "state": state, **values}


def render_string(name: str, **values):
    return templates.env.get_template(name).render(state=state, **values)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not state.hardware_ready:
        await run_in_threadpool(state.refresh_hardware)
    if not state.models:
        try:
            await run_in_threadpool(state.scan_models)
        except ValueError:
            pass
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context(
            request,
            hardware=state.hardware_snapshot(),
            models=state.models_snapshot(),
            model=state.model_snapshot(),
            parameters=state.parameter_snapshot(),
            process=state.process_snapshot(),
            snn=await run_in_threadpool(state.snn_snapshot),
            download=state.download_snapshot(),
        ),
    )


@app.post("/partials/hardware", response_class=HTMLResponse)
async def hardware(request: Request):
    data = await run_in_threadpool(state.refresh_hardware)
    return templates.TemplateResponse(
        request=request, name="partials/hardware.html",
        context=context(request, hardware=data),
    )


@app.post("/system/swap", response_class=HTMLResponse)
async def system_swap(request: Request):
    form = await form_values(request)
    action = form.get("action", "create")
    try:
        size = int(form.get("size_gib", "0") or 0)
        data = await run_in_threadpool(state.configure_nvme_swap, action, size)
        message = (
            "Swap NVMe dinâmico aplicado."
            if action == "create" else
            "Swap NVMe do Crono Matrix removido."
        )
        error = ""
    except (OSError, ValueError) as exc:
        data = state.hardware_snapshot()
        message, error = "", str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/hardware.html",
        context=context(request, hardware=data, message=message, error=error),
    )


@app.post("/system/swap/auto", response_class=HTMLResponse)
async def system_swap_auto(request: Request):
    form = await form_values(request)
    enabled = str(form.get("enabled", "n")).lower() in {"1", "true", "y", "on"}
    try:
        data = await run_in_threadpool(state.set_auto_nvme_swap, enabled)
        message, error = (
            "Crescimento automático do swap ativado."
            if enabled else
            "Crescimento automático do swap desativado."
        ), ""
    except (OSError, ValueError) as exc:
        data = state.hardware_snapshot()
        message, error = "", str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/hardware.html",
        context=context(request, hardware=data, message=message, error=error),
    )


@app.post("/partials/models", response_class=HTMLResponse)
async def models(request: Request):
    form = await form_values(request)
    try:
        # The update checker polls this endpoint every second while its worker
        # hashes large GGUFs. Re-scanning GGUF headers on every poll would add
        # needless disk work and could disturb a long-running model load.
        models_dir = form.get("models_dir", "")
        if models_dir or not state.models:
            rows = await run_in_threadpool(state.scan_models, models_dir)
        else:
            rows = state.models_snapshot()
        error = ""
    except ValueError as exc:
        rows = state.models_snapshot()
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/models.html",
        context=context(
            request, models=rows, error=error,
            update_status=state.model_update_snapshot(),
        ),
    )


@app.post("/models/verify-updates", response_class=HTMLResponse)
async def verify_model_updates(request: Request):
    """Start a non-blocking origin/hash check for every local GGUF."""
    try:
        rows = await run_in_threadpool(state.start_model_update_check)
        error = ""
    except ValueError as exc:
        rows = state.models_snapshot()
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/models.html",
        context=context(
            request, models=rows, error=error,
            update_status=state.model_update_snapshot(),
        ),
    )


@app.post("/partials/configure-paths", response_class=HTMLResponse)
async def configure_paths(request: Request):
    form = await form_values(request)
    try:
        await run_in_threadpool(
            state.configure_paths,
            form.get("llama_cpp_dir", ""),
            form.get("models_dir", ""),
        )
        error = ""
    except ValueError as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/configuration_response.html",
        context=context(
            request, error=error, models=state.models_snapshot(),
            model=state.model_snapshot(), parameters=state.parameter_snapshot(),
        ),
    )


@app.post("/partials/select-model", response_class=HTMLResponse)
async def select_model(request: Request):
    form = await form_values(request)
    try:
        await run_in_threadpool(state.select_model, form.get("model_id", ""))
        error = ""
    except ValueError as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/select_response.html",
        context=context(
            request, error=error, models=state.models_snapshot(),
            model=state.model_snapshot(), parameters=state.parameter_snapshot(),
        ),
    )


@app.post("/partials/recalculate-memory", response_class=HTMLResponse)
async def recalculate_memory(request: Request):
    form = await form_values(request)
    try:
        await run_in_threadpool(state.recalculate_memory, form)
        error = ""
    except ValueError as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/recalculate_response.html",
        context=context(
            request, error=error, model=state.model_snapshot(),
            parameters=state.parameter_snapshot(),
        ),
    )


@app.post("/partials/preview", response_class=HTMLResponse)
async def preview(request: Request):
    form = await form_values(request)
    try:
        _final, _cmd, display = state.preview_command(form)
        error = ""
    except ValueError as exc:
        display = ""
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/command.html",
        context=context(request, command=display, error=error),
    )


@app.post("/server/start", response_class=HTMLResponse)
async def start_server(request: Request):
    form = await form_values(request)
    try:
        process = await run_in_threadpool(state.start_server, form)
        message, error = "Servidor iniciado.", ""
    except Exception as exc:
        process = state.process_snapshot()
        message, error = "", str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/action_response.html",
        context=context(
            request, process=process, model=state.model_snapshot(),
            parameters=state.parameter_snapshot(),
            message=message, error=error,
        ),
    )


@app.post("/server/stop", response_class=HTMLResponse)
async def stop_server(request: Request):
    process = await run_in_threadpool(state.stop_server)
    return templates.TemplateResponse(
        request=request, name="partials/action_response.html",
        context=context(
            request, process=process, model=state.model_snapshot(),
            parameters=state.parameter_snapshot(),
            message="Servidor encerrado.", error="",
        ),
    )


@app.post("/agent/toggle", response_class=HTMLResponse)
async def agent_toggle(request: Request):
    form = await form_values(request)
    enabled = str(form.get("enabled", "")).lower() in {"1", "true", "y", "on"}
    try:
        # The toggle method may return only the compatibility payload when no
        # restart is needed. Always render the complete process snapshot so
        # HTMX receives the same shape as the SSE and server-start routes.
        await run_in_threadpool(state.set_agent_global, enabled)
        process = await run_in_threadpool(state.process_snapshot)
        message = (
            (
                "Modo universal preparado: será aplicado ao próximo servidor iniciado."
                if not process.get("running") else
                "Modo universal ativado: qualquer cliente OpenAI-compatible pode usar o modelo do Crono Matrix."
            )
            if enabled else
            "Modo universal desativado; a configuração de clientes não foi alterada."
        )
        error = ""
    except (OSError, ValueError) as exc:
        process = state.process_snapshot()
        message, error = "", str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/action_response.html",
        context=context(
            request, process=process, model=state.model_snapshot(),
            parameters=state.parameter_snapshot(),
            message=message, error=error,
        ),
    )


@app.get("/api/snn")
async def snn_api():
    return await run_in_threadpool(state.snn_snapshot)


@app.get("/partials/snn", response_class=HTMLResponse)
async def snn_partial(request: Request):
    return templates.TemplateResponse(
        request=request, name="partials/snn.html",
        context=context(request, snn=await run_in_threadpool(state.snn_snapshot)),
    )


@app.post("/snn/toggle", response_class=HTMLResponse)
async def snn_toggle(request: Request):
    form = await form_values(request)
    enabled = str(form.get("enabled", "")).lower() in {"1", "true", "y", "on"}
    try:
        snapshot = await run_in_threadpool(state.set_snn_enabled, enabled)
        error = ""
    except (OSError, ValueError) as exc:
        snapshot = await run_in_threadpool(state.snn_snapshot)
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/snn.html",
        context=context(request, snn=snapshot, snn_error=error),
    )


@app.post("/launcher/stop", response_class=HTMLResponse)
async def stop_launcher(request: Request):
    client_host = request.client.host if request.client else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = client_host == "localhost"
    if not is_loopback:
        return HTMLResponse("Encerramento permitido apenas por loopback.", status_code=403)

    async def terminate_after_response():
        await asyncio.sleep(0.25)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(terminate_after_response())
    return HTMLResponse("Launcher encerrando...")


@app.get("/partials/hf-search", response_class=HTMLResponse)
async def hf_search(request: Request, q: str = ""):
    if len(q.strip()) < 2:
        results, error = [], "Digite ao menos dois caracteres."
    else:
        try:
            results = await run_in_threadpool(state.search_hf, q)
            error = ""
        except Exception as exc:
            results, error = [], str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/hf_results.html",
        context=context(request, results=results, error=error, query=q),
    )


@app.get("/partials/hf-radar", response_class=HTMLResponse)
async def hf_radar(request: Request, force: bool = False):
    radar = await run_in_threadpool(state.refresh_hf_radar, force)
    return templates.TemplateResponse(
        request=request, name="partials/hf_radar.html",
        context=context(request, radar=radar),
    )


@app.post("/hf/radar/preferences", response_class=HTMLResponse)
async def hf_radar_preferences(request: Request):
    form = await form_values(request)
    enabled = str(form.get("enabled", "")).lower() in {"1", "true", "y", "yes", "on"}
    radar = await run_in_threadpool(
        state.set_hf_radar_preferences,
        form.get("watchlist", ""),
        enabled,
    )
    return templates.TemplateResponse(
        request=request, name="partials/hf_radar.html",
        context=context(request, radar=radar),
    )


@app.post("/hf/radar/read", response_class=HTMLResponse)
async def hf_radar_read(request: Request):
    radar = await run_in_threadpool(state.mark_hf_radar_read)
    return templates.TemplateResponse(
        request=request, name="partials/hf_radar.html",
        context=context(request, radar=radar),
    )


@app.get("/partials/hf-detail", response_class=HTMLResponse)
async def hf_detail(request: Request, repo_id: str):
    try:
        detail = await run_in_threadpool(state.hf_details, repo_id)
        error = ""
    except Exception as exc:
        detail, error = None, str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/hf_detail.html",
        context=context(request, detail=detail, error=error),
    )


@app.post("/hf/download", response_class=HTMLResponse)
async def hf_download(request: Request):
    form = await form_values(request)
    try:
        download = state.start_download(form.get("repo_id", ""), form.get("filename", ""))
        error = ""
    except Exception as exc:
        download, error = state.download_snapshot(), str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/download.html",
        context=context(request, download=download, error=error),
    )


@app.get("/partials/download", response_class=HTMLResponse)
async def download_status(request: Request):
    return templates.TemplateResponse(
        request=request, name="partials/download.html",
        context=context(request, download=state.download_snapshot(), error=""),
    )


@app.post("/hf/download/cancel", response_class=HTMLResponse)
async def cancel_download(request: Request):
    download = state.cancel_download()
    return templates.TemplateResponse(
        request=request, name="partials/download.html",
        context=context(request, download=download, error=""),
    )


@app.get("/events")
async def events(request: Request):
    async def stream():
        sequence = 0
        previous_process = None
        previous_snn = None
        while not await request.is_disconnected():
            process = state.process_snapshot()
            encoded = json.dumps(process, sort_keys=True, default=str)
            if encoded != previous_process:
                markup = render_string("partials/server.html", process=process)
                markup = markup.replace("\n", " ")
                yield f"event: server\ndata: {markup}\n\n"
                previous_process = encoded
            snn = await run_in_threadpool(state.snn_snapshot)
            encoded_snn = json.dumps(snn, sort_keys=True, default=str)
            if encoded_snn != previous_snn:
                markup = render_string("partials/snn.html", snn=snn)
                markup = markup.replace("\n", " ")
                yield f"event: snn\ndata: {markup}\n\n"
                previous_snn = encoded_snn
            for item in state.logs_after(sequence):
                sequence = item["seq"]
                line = html.escape(item["line"].rstrip("\n")) or " "
                level = html.escape(item["level"])
                yield f'event: log\ndata: <div class="log-line {level}">{line}</div>\n\n'
            yield ": heartbeat\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/eval", response_class=HTMLResponse)
async def eval_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="eval.html",
        context=context(request, eval_state=eval_runner.snapshot()),
    )


@app.get("/partials/eval-status", response_class=HTMLResponse)
async def eval_status(request: Request):
    return templates.TemplateResponse(
        request=request, name="partials/eval_status.html",
        context=context(request, eval_state=eval_runner.snapshot()),
    )


@app.get("/partials/eval-dashboard", response_class=HTMLResponse)
async def eval_dashboard(request: Request):
    data, ts = eval_runner.get_dashboard_data()
    return templates.TemplateResponse(
        request=request, name="partials/eval_dashboard.html",
        context=context(request, eval_data=data, eval_ts=ts),
    )


@app.get("/eval/data")
async def eval_data(request: Request):
    data, _ts = eval_runner.get_dashboard_data()
    if data is None:
        return HTMLResponse(content="{}", media_type="application/json")
    return data


@app.get("/eval/export")
async def eval_export():
    try:
        filename, content = await run_in_threadpool(eval_runner.academic_export)
    except ValueError as exc:
        return HTMLResponse(str(exc), status_code=404)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/eval/runs/delete")
async def eval_delete_run(request: Request):
    form = await form_values(request)
    try:
        return await run_in_threadpool(eval_runner.delete_run, form.get("checkpoint_id", ""))
    except ValueError as exc:
        return Response(str(exc), status_code=400, media_type="text/plain")


@app.post("/eval/start", response_class=HTMLResponse)
async def eval_start(request: Request):
    form = await form_values(request)
    try:
        eval_runner.start_run(
            axes_filter=form.get("axes", ""),
            repeats=form.get("repeats", "1"),
            seed=form.get("seed", "0"),
            scale=form.get("scale", "auto"),
            mode=form.get("mode", "auto"),
            api_url=form.get("eval_url", ""),
            reasoning_effort=form.get("reasoning_effort", "default"),
            reasoning_budget=form.get("reasoning_budget", "auto"),
            sampling=form.get("sampling", "server"),
            temperature=form.get("temperature", "0.6"),
            top_k=form.get("top_k", "20"),
            top_p=form.get("top_p", "0.95"),
            min_p=form.get("min_p", "0.05"),
            repeat_penalty=form.get("repeat_penalty", "1.0"),
            max_tokens=form.get("max_tokens", "16384"),
            timeout=form.get("timeout", "300"),
            xctx_scale=form.get("xctx_scale", "1.0"),
            os_filter=form.get("os_filter", "all"),
            judge_url=form.get("judge_url", ""),
            judge_model=form.get("judge_model", ""),
            runtime_context=state.evaluation_context(),
        )
        error = ""
    except ValueError as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request=request, name="partials/eval_status.html",
        context=context(request, eval_state=eval_runner.snapshot(), eval_error=error),
    )


@app.post("/eval/stop", response_class=HTMLResponse)
async def eval_stop(request: Request):
    eval_runner.stop_run()
    await asyncio.sleep(0.5)
    return templates.TemplateResponse(
        request=request, name="partials/eval_status.html",
        context=context(request, eval_state=eval_runner.snapshot()),
    )


@app.get("/eval/events")
async def eval_events(request: Request):
    async def stream():
        sequence = 0
        previous = None
        while not await request.is_disconnected():
            snap = eval_runner.snapshot()
            encoded = json.dumps(snap, sort_keys=True, default=str)
            if encoded != previous:
                markup = render_string("partials/eval_status_bar.html", eval_state=snap)
                markup = markup.replace("\n", " ")
                yield f"event: status\ndata: {markup}\n\n"
                yield f"event: progress\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n"
                previous = encoded
            for item in eval_runner.logs_after(sequence):
                sequence = item["seq"]
                line = html.escape(item["line"].rstrip("\n")) or " "
                level = html.escape(item["level"])
                yield f"event: log\ndata: <div class=\"log-line {level}\">{line}</div>\n\n"
            yield ": heartbeat\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

#!/usr/bin/env python3
"""Interface desktop moderna do Crono Matrix.

Esta camada e deliberadamente fina: descoberta de hardware/modelos, calculo de
parametros, validacao e gerenciamento do llama-server pertencem ao
``LauncherWebState``. Assim, desktop e web enviam exatamente o mesmo comando ao
llama.cpp.
"""

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable

try:
    import customtkinter as ctk
except ImportError as exc:  # pragma: no cover - mensagem para instalacao real
    raise SystemExit(
        "CustomTkinter nao esta instalado. Execute: "
        ".venv/bin/pip install -r requirements-gui.txt"
    ) from exc

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependencia declarada para a GUI
    Image = None

from web.services import LauncherWebState, EvalRunner
from launch_model_core import SPECULATIVE_TYPES


ROOT = Path(__file__).resolve().parent
DONATION_QR_PATH = ROOT / "web" / "static" / "assets" / "donation-qrcode.jpeg"
BUNDLED_MCP_AVAILABLE = (ROOT / "mcp-crono-matrix" / "native_server.mjs").is_file()
BG = "#050b08"
SIDEBAR = "#08130d"
CARD = "#0b1811"
CARD_ALT = "#0e2017"
BORDER = "#1d4b32"
GREEN = "#42f58d"
GREEN_DARK = "#148c4c"
GREEN_HOVER = "#22b966"
TEXT = "#d7eee0"
MUTED = "#7fa88e"
CYAN = "#62d9e8"
RED = "#ff6474"
AMBER = "#efc56b"
MONO = ("DejaVu Sans Mono", 12)


def human_size(size: float) -> str:
    return f"{size:.2f} GiB"


def format_bytes(size: Any) -> str:
    try:
        value = max(float(size or 0), 0.0)
    except (TypeError, ValueError):
        value = 0.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return "0.0 B"


def hf_download_view(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza o estado compartilhado de download para as duas UIs."""
    source = snapshot if isinstance(snapshot, dict) else {}
    state = str(source.get("state") or "idle").lower()
    try:
        downloaded = max(int(source.get("downloaded") or 0), 0)
        total = max(int(source.get("total") or 0), 0)
        speed = max(float(source.get("speed") or 0), 0.0)
    except (TypeError, ValueError):
        downloaded, total, speed = 0, 0, 0.0
    progress = min(downloaded / total, 1.0) if total else 0.0
    labels = {
        "idle": "Aguardando seleção",
        "running": "BAIXANDO",
        "cancelling": "CANCELANDO",
        "cancelled": "CANCELADO",
        "done": "CONCLUÍDO",
        "error": "ERRO",
    }
    return {
        "state": state,
        "label": labels.get(state, state.upper()),
        "filename": str(source.get("filename") or ""),
        "downloaded": downloaded,
        "total": total,
        "speed": speed,
        "progress": progress,
        "percent": progress * 100.0,
        "error": str(source.get("error") or ""),
        "paths": list(source.get("paths") or []),
    }


def compact_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value or "—")


def adaptive_window_metrics(screen_w: int, screen_h: int) -> dict[str, int | float]:
    """Retorna geometria/escala seguras para a resolução lógica da tela."""
    screen_w = max(int(screen_w), 800)
    screen_h = max(int(screen_h), 600)
    if screen_h <= 768 or screen_w <= 1366:
        scale = 0.78
    elif screen_h <= 900 or screen_w <= 1600:
        scale = 0.86
    elif screen_h <= 1080:
        scale = 0.94
    else:
        scale = 1.0
    width = min(1480, max(920, int(screen_w * 0.91)), screen_w - 24)
    height = min(920, max(620, int(screen_h * 0.84)), screen_h - 54)
    min_width = min(width, 1040, max(760, int(screen_w * 0.72)))
    min_height = min(height, 660, max(520, int(screen_h * 0.66)))
    return {
        "screen_w": screen_w, "screen_h": screen_h, "scale": scale,
        "width": width, "height": height,
        "min_width": min_width, "min_height": min_height,
        "x": max((screen_w - width) // 2, 0),
        "y": max((screen_h - height) // 2, 0),
    }


def vram_plan_action(
    planned_free_mb: int, current_free_mb: int, threshold_mb: int = 512,
) -> str:
    """Classify whether a launch profile is still valid for current VRAM."""
    planned = max(int(planned_free_mb), 0)
    current = max(int(current_free_mb), 0)
    threshold = max(int(threshold_mb), 1)
    if planned == 0:
        return "keep"
    if current - planned >= threshold:
        return "rebuild"
    if planned - current >= threshold:
        return "block"
    return "keep"


def memory_guard_view(guard: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza a telemetria produzida pelo guard C99 para a interface."""
    source = guard if isinstance(guard, dict) else {}

    def integer(key: str, default: int = 0) -> int:
        try:
            return max(int(source.get(key, default) or 0), 0)
        except (TypeError, ValueError):
            return max(default, 0)

    available_mb = integer("available_mb")
    current_mb = integer("current_mb")
    trigger_mb = integer("trigger_mb", 1536)
    error = str(source.get("error") or "").strip()
    high_mb = integer("memory_high_mb")
    return {
        "available_mb": available_mb,
        "current_mb": current_mb,
        "reserve_mb": integer("reserve_mb", 1024),
        "trigger_mb": trigger_mb,
        "pressure_count": integer("pressure_count"),
        "last_action": str(source.get("last_action") or "aguardando servidor"),
        "error": error,
        "scope_unit": str(source.get("scope_unit") or "—"),
        "scope_phase": str(source.get("scope_phase") or "idle"),
        "memory_high": "max" if high_mb == 0 else f"{high_mb} MiB",
        "scope_headroom_mb": integer("scope_headroom_mb"),
        "pressure": bool(error or (available_mb and available_mb < trigger_mb)),
    }


def decoration_geometry_pulse(raw_geometry: str) -> tuple[str, str] | None:
    """Cria um resize mínimo que força o KWin a recalcular a decoração."""
    match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", str(raw_geometry))
    if not match:
        return None
    width, height, x_pos, y_pos = match.groups()
    width_value = int(width)
    if width_value < 1:
        return None
    original = f"{width_value}x{int(height)}{x_pos}{y_pos}"
    pulse = f"{width_value + 1}x{int(height)}{x_pos}{y_pos}"
    return pulse, original


class CronoDesktop(ctk.CTk):
    """CustomTkinter shell backed by the production launcher service."""

    MEMORY_FIELDS = {
        "ctx", "cache_k", "cache_v", "kv_offload", "batch", "ubatch",
        "omni", "swa_full", "spec_type", "parallel", "fit_target", "fit",
        "cache_ram", "ctx_checkpoints", "fit_ctx",
    }
    MEMORY_PLAN_FIELDS = MEMORY_FIELDS | {
        "ctx", "ngl", "flash", "kv_offload", "kv_unified", "threads",
        "threads_batch", "n_cpu_moe", "n_cpu_ffn", "load_mode",
        "tensor_read_lazy", "numa", "device", "mmproj_offload",
    }
    MEMORY_DERIVED_FIELDS = MEMORY_PLAN_FIELDS | {
        "split_mode", "repack", "direct_io", "no_host", "cache_reuse",
        "poll", "mlock", "no_mmap", "fit_ctx", "mtmd_batch_max",
        "image_min_tokens", "image_max_tokens", "cpu_moe",
    }

    def __init__(self, state: LauncherWebState | None = None, eval_runner: EvalRunner | None = None):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        super().__init__(fg_color=BG)
        # ``CTk.state()`` é um método usado internamente pelo rastreador de
        # DPI. Não sombreie esse nome com o serviço da aplicação.
        self.backend = state or LauncherWebState()
        self.eval_runner = eval_runner or EvalRunner()
        self.title("Crono Matrix · Controle local llama.cpp")
        self._configure_adaptive_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.vars: dict[str, ctk.StringVar] = {}
        self.fields: dict[str, Any] = {}
        self.pages: dict[str, ctk.CTkFrame] = {}
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.model_buttons: list[Any] = []
        self.async_events: queue.Queue = queue.Queue()
        self.log_sequence = 0
        self._eval_log_seq = 0
        self._updating_form = False
        self._recalc_job: str | None = None
        self._busy = 0
        self._closing = False
        self._manual_fields: set[str] = set()
        self._profile_vram_free_mb = 0
        self._last_memory_guard_view: tuple[Any, ...] | None = None
        self._last_process_render = ""
        self._last_eval_render = ""
        self._last_eval_dashboard_render: tuple[Any, ...] | None = None
        self._last_radar_render = ""
        self._last_download_render = ""
        self._download_completion_seen = ""
        self._radar_selected_repo = ""
        self._last_snn_render = ""
        self._donation_qr_thumb = None
        self._donation_qr_dialog = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_shell()
        self._build_models_page()
        self._build_inference_page()
        self._build_process_page()
        self._build_system_page()
        self._build_eval_page()
        self._build_radar_page()
        self.show_page("models")

        # Em algumas versões do KWin a decoração inicial de uma janela Tk
        # omite o botão fechar até o primeiro maximizar/restaurar. Um resize
        # transitório de 1 px após o mapeamento atualiza a moldura sem alterar
        # o tamanho final nem exigir ação do usuário.
        self.after(220, self._refresh_window_decorations)

        self.after(80, self._drain_async)
        self.after(300, self._poll_runtime)
        self._run_task(self._initial_load, self._initial_ready, "Detectando hardware e modelos…")

    def _configure_adaptive_window(self) -> None:
        """Dimensiona a UI pela tela real sem forçar maximização.

        O Tk informa a resolução lógica já corrigida pelo compositor. A escala
        abaixo reduz controles em telas baixas (notebooks/TVs 720p–900p) e
        preserva o tamanho normal em Full HD ou superior.
        """
        metrics = adaptive_window_metrics(self.winfo_screenwidth(), self.winfo_screenheight())
        if sys.platform.startswith("linux"):
            try:
                self.attributes("-type", "normal")
            except Exception:
                # Nem todo Tk/XWayland expõe o atributo EWMH ``-type``.
                pass
        ctk.set_widget_scaling(float(metrics["scale"]))
        self.geometry(
            f"{metrics['width']}x{metrics['height']}+{metrics['x']}+{metrics['y']}"
        )
        self.minsize(int(metrics["min_width"]), int(metrics["min_height"]))
        self.ui_scale = float(metrics["scale"])
        self.screen_geometry = f"{metrics['screen_w']}×{metrics['screen_h']}"

    def _refresh_window_decorations(self) -> None:
        """Força uma única atualização segura da moldura no KWin/XWayland."""
        if self._closing or not sys.platform.startswith("linux"):
            return
        try:
            if self.state() != "normal":
                return
            raw = str(self.tk.call("wm", "geometry", self._w))
            geometry = decoration_geometry_pulse(raw)
            if not geometry:
                return
            pulse, original = geometry
            self.tk.call("wm", "geometry", self._w, pulse)
            self.after(70, lambda: self._restore_decoration_geometry(pulse, original))
        except Exception:
            # Decoração é responsabilidade do WM; falhar aqui não pode impedir
            # o uso da interface nem alterar o plano de inferência.
            return

    def _restore_decoration_geometry(self, pulse: str, original: str) -> None:
        if self._closing:
            return
        try:
            current = str(self.tk.call("wm", "geometry", self._w))
            if current == pulse:
                self.tk.call("wm", "geometry", self._w, original)
        except Exception:
            return

    # ------------------------------------------------------------------ shell
    def _build_sidebar(self) -> None:
        side = ctk.CTkFrame(self, width=238, corner_radius=0, fg_color=SIDEBAR)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        # As opções ocupam as linhas 3–8. O espaço flexível precisa ficar em
        # uma linha própria; dar peso à linha 8 afastava o Radar da Avaliação.
        side.grid_rowconfigure(9, weight=1)

        ctk.CTkLabel(
            side, text="C://", text_color=GREEN,
            font=ctk.CTkFont(family="DejaVu Sans Mono", size=30, weight="bold"),
        ).grid(row=0, column=0, padx=24, pady=(25, 0), sticky="w")
        ctk.CTkLabel(
            side, text="CRONO MATRIX", text_color=TEXT,
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=1, column=0, padx=24, pady=(0, 2), sticky="w")
        ctk.CTkLabel(
            side, text="LOCAL CONTROL PLANE", text_color=MUTED,
            font=ctk.CTkFont(family="DejaVu Sans Mono", size=10),
        ).grid(row=2, column=0, padx=24, pady=(0, 28), sticky="w")

        entries = (
            ("models", "01  MODELOS"),
            ("inference", "02  INFERÊNCIA"),
            ("process", "03  PROCESSO"),
            ("system", "04  SISTEMA"),
            ("eval", "05  AVALIAÇÃO"),
            ("radar", "06  RADAR · ONLINE"),
        )
        for row, (key, label) in enumerate(entries, start=3):
            button = ctk.CTkButton(
                side, text=label, anchor="w", height=42, corner_radius=7,
                fg_color="transparent", hover_color=CARD_ALT, text_color=MUTED,
                font=ctk.CTkFont(family="DejaVu Sans Mono", size=12, weight="bold"),
                command=lambda selected=key: self.show_page(selected),
            )
            button.grid(row=row, column=0, padx=14, pady=3, sticky="ew")
            self.nav_buttons[key] = button

        support = ctk.CTkFrame(side, fg_color=CARD, corner_radius=10, border_width=1, border_color=BORDER)
        support.grid(row=10, column=0, padx=16, pady=14, sticky="sew")
        ctk.CTkLabel(
            support, text="APOIE O DESENVOLVIMENTO", text_color=GREEN,
            font=ctk.CTkFont(family="DejaVu Sans Mono", size=10, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 5))

        support_body = ctk.CTkFrame(support, fg_color="transparent")
        support_body.pack(fill="x", padx=10)
        if Image is not None and DONATION_QR_PATH.is_file():
            with Image.open(DONATION_QR_PATH) as source:
                qr_source = source.convert("RGB").copy()
            self._donation_qr_thumb = ctk.CTkImage(
                light_image=qr_source, dark_image=qr_source, size=(72, 87),
            )
            ctk.CTkButton(
                support_body, image=self._donation_qr_thumb, text="", width=80, height=95,
                fg_color="#ffffff", hover_color="#d9f5e2", corner_radius=5,
                command=self._show_donation_qr,
            ).pack(side="left", padx=(0, 8), pady=(0, 7))
            support_message = "PIX opcional\nClique para ampliar"
        else:
            support_message = "QR indisponível\nConsulte docs/SUPPORT.md"
        ctk.CTkLabel(
            support_body, text=support_message, text_color=TEXT, justify="left",
            font=ctk.CTkFont(size=10),
        ).pack(side="left", anchor="center")
        ctk.CTkLabel(
            support,
            text=f"● DESKTOP LOCAL · TELA {self.screen_geometry}",
            text_color="#507660", font=ctk.CTkFont(family="DejaVu Sans Mono", size=8),
        ).pack(anchor="w", padx=12, pady=(0, 8))

    def _show_donation_qr(self) -> None:
        """Abre o QR local em tamanho legível, sem acessar a rede."""
        if Image is None or not DONATION_QR_PATH.is_file():
            messagebox.showerror("Apoie o Crono Matrix", "Arquivo do QR Code não encontrado.")
            return
        if self._donation_qr_dialog is not None and self._donation_qr_dialog.winfo_exists():
            self._donation_qr_dialog.focus_force()
            return

        dialog = ctk.CTkToplevel(self, fg_color=BG)
        self._donation_qr_dialog = dialog
        dialog.title("Apoie o desenvolvimento · Crono Matrix")
        dialog.geometry("440x570")
        dialog.resizable(False, False)
        dialog.transient(self)

        ctk.CTkLabel(
            dialog, text="APOIE O CRONO MATRIX", text_color=GREEN,
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(22, 4))
        ctk.CTkLabel(
            dialog, text="Sua contribuição ajuda a manter o desenvolvimento.",
            text_color=MUTED, font=ctk.CTkFont(size=12),
        ).pack(pady=(0, 12))

        with Image.open(DONATION_QR_PATH) as source:
            qr_source = source.convert("RGB").copy()
        self._donation_qr_dialog_image = ctk.CTkImage(
            light_image=qr_source, dark_image=qr_source, size=(306, 369),
        )
        ctk.CTkLabel(
            dialog, text="", image=self._donation_qr_dialog_image,
            fg_color="#ffffff", corner_radius=10,
        ).pack(padx=28, pady=4)
        ctk.CTkLabel(
            dialog, text="Doação voluntária via PIX · nenhum recurso é bloqueado",
            text_color=MUTED, font=ctk.CTkFont(family="DejaVu Sans Mono", size=9),
        ).pack(pady=(8, 4))
        ctk.CTkButton(
            dialog, text="FECHAR", width=120, fg_color=GREEN_DARK,
            hover_color=GREEN_HOVER, command=dialog.destroy,
        ).pack(pady=(4, 16))

    def _build_shell(self) -> None:
        shell = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        shell.grid(row=0, column=1, sticky="nsew")
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(1, weight=1)
        self.shell = shell

        top = ctk.CTkFrame(shell, height=78, corner_radius=0, fg_color="#07110c")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="MODELO ATIVO", text_color=MUTED, font=MONO).grid(
            row=0, column=0, padx=(25, 10), pady=(15, 0), sticky="w"
        )
        self.active_model_label = ctk.CTkLabel(
            top, text="Nenhum modelo selecionado", text_color=TEXT,
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        )
        self.active_model_label.grid(row=1, column=0, columnspan=2, padx=25, pady=(0, 12), sticky="ew")
        self.runtime_badge = ctk.CTkLabel(
            top, text="◇  IDLE", text_color=GREEN, fg_color=CARD,
            corner_radius=14, width=130, height=34, font=MONO,
        )
        self.runtime_badge.grid(row=0, column=2, rowspan=2, padx=24, pady=20)

        self.page_host = ctk.CTkFrame(shell, fg_color=BG, corner_radius=0)
        self.page_host.grid(row=1, column=0, sticky="nsew", padx=18, pady=16)
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)

        footer = ctk.CTkFrame(shell, height=32, fg_color="#07110c", corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(footer, text="Inicializando…", text_color=MUTED, font=MONO)
        self.status_label.grid(row=0, column=0, padx=18, pady=5, sticky="w")
        self.busy_bar = ctk.CTkProgressBar(footer, width=150, height=5, progress_color=GREEN)
        self.busy_bar.grid(row=0, column=1, padx=18)
        self.busy_bar.set(0)

    def _new_page(self, name: str) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.page_host, fg_color=BG, corner_radius=0)
        page.grid(row=0, column=0, sticky="nsew")
        self.pages[name] = page
        return page

    def show_page(self, name: str) -> None:
        page = self.pages.get(name)
        if page:
            page.tkraise()
        for key, button in self.nav_buttons.items():
            button.configure(
                fg_color=CARD_ALT if key == name else "transparent",
                text_color=GREEN if key == name else MUTED,
            )

    @staticmethod
    def _title(parent: Any, title: str, subtitle: str) -> None:
        ctk.CTkLabel(parent, text=title, text_color=TEXT, font=ctk.CTkFont(size=24, weight="bold")).pack(
            anchor="w", padx=4, pady=(0, 2)
        )
        ctk.CTkLabel(parent, text=subtitle, text_color=MUTED, font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=4, pady=(0, 14)
        )

    @staticmethod
    def _card(parent: Any) -> ctk.CTkFrame:
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, border_width=1, border_color=BORDER)

    # --------------------------------------------------------------- modelos
    def _build_models_page(self) -> None:
        page = self._new_page("models")
        page.grid_columnconfigure(0, weight=3)
        page.grid_columnconfigure(1, weight=2)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._title(head, "Modelos locais", "Descoberta GGUF e perfil automático orientado ao hardware atual")

        left = self._card(page)
        left.grid(row=1, column=0, padx=(0, 8), sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)

        paths = ctk.CTkFrame(left, fg_color="transparent")
        paths.grid(row=0, column=0, padx=14, pady=14, sticky="ew")
        paths.grid_columnconfigure(1, weight=1)
        cfg = self.backend.configuration_snapshot()
        self.llama_path_var = ctk.StringVar(value=cfg["llama_cpp_dir"])
        self.models_path_var = ctk.StringVar(value=cfg["models_dir"])
        self._path_row(paths, 0, "llama.cpp", self.llama_path_var, self._choose_llama)
        self._path_row(paths, 1, "Modelos", self.models_path_var, self._choose_models)
        ctk.CTkButton(
            paths, text="APLICAR E VARRER", height=34, fg_color=GREEN_DARK,
            hover_color=GREEN_HOVER, command=self._apply_paths,
        ).grid(row=2, column=1, padx=7, pady=(8, 0), sticky="e")
        ctk.CTkButton(
            paths, text="VERIFICAR ATUALIZAÇÕES · ONLINE", height=34, fg_color=CARD_ALT,
            hover_color=GREEN_DARK, command=self._verify_model_updates,
        ).grid(row=3, column=1, padx=7, pady=(4, 0), sticky="e")
        self.update_status_label = ctk.CTkLabel(
            paths, text="", text_color=MUTED, font=ctk.CTkFont(size=10),
        )
        self.update_status_label.grid(row=4, column=1, padx=7, pady=(2, 0), sticky="e")

        separator = ctk.CTkFrame(left, height=1, fg_color=BORDER)
        separator.grid(row=1, column=0, sticky="ew", padx=14)
        self.model_list = ctk.CTkScrollableFrame(left, fg_color="transparent", corner_radius=0)
        self.model_list.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        self.model_list.grid_columnconfigure(0, weight=1)

        right = self._card(page)
        right.grid(row=1, column=1, padx=(8, 0), sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        self.model_detail = ctk.CTkTextbox(
            right, fg_color="transparent", border_width=0, text_color=TEXT,
            font=MONO, wrap="word", activate_scrollbars=True,
        )
        self.model_detail.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        right.grid_rowconfigure(0, weight=1)
        self._replace_text(self.model_detail, "Selecione um GGUF para ler os metadados e calcular o perfil ideal.\n")

    def _path_row(self, parent: Any, row: int, label: str, variable: Any, command: Callable) -> None:
        ctk.CTkLabel(parent, text=label, text_color=MUTED, font=MONO).grid(
            row=row, column=0, padx=(0, 8), pady=5, sticky="w"
        )
        ctk.CTkEntry(
            parent, textvariable=variable, fg_color=BG, border_color=BORDER,
            text_color=TEXT, font=ctk.CTkFont(family="DejaVu Sans Mono", size=11),
        ).grid(row=row, column=1, padx=7, pady=5, sticky="ew")
        ctk.CTkButton(
            parent, text="…", width=38, fg_color=CARD_ALT, hover_color=GREEN_DARK,
            command=command,
        ).grid(row=row, column=2, padx=(7, 0), pady=5)

    # ------------------------------------------------------------- inferencia
    def _build_inference_page(self) -> None:
        page = self._new_page("inference")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        self._title(head, "Inferência", "Controles efetivos do llama-server; alterações de memória recalculam o plano")

        body = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure((0, 1), weight=1)

        memory = self._section(body, "GPU, CONTEXTO E KV", 0, 0)
        self._field(memory, "ctx", "Janela de contexto", 0, 0)
        self._field(memory, "ngl", "Camadas GPU", 0, 1, ["auto", "all"])
        self._field(memory, "cache_k", "Cache K", 1, 0, ["f32", "f16", "bf16", "q8_0", "q5_1", "q4_1", "q4_0"])
        self._field(memory, "cache_v", "Cache V", 1, 1, ["f32", "f16", "bf16", "q8_0", "q5_1", "q4_1", "q4_0"])
        self._field(memory, "parallel", "Slots paralelos", 2, 0)
        self._field(memory, "fit_target", "Reserva VRAM (MiB)", 2, 1)
        self._switch(memory, "fit", "Fit / planejamento automático", 3, 0)
        self._field(memory, "flash", "Flash Attention", 3, 1, ["auto", "y", "n"])
        self._field(memory, "kv_offload", "KV offload", 4, 0, ["y", "n"])
        self._switch(memory, "kv_unified", "KV unificado", 4, 1)
        self._switch(memory, "swa_full", "SWA full", 5, 0)
        self._field(memory, "cache_ram", "Cache prompt RAM (MiB)", 5, 1)
        self._field(memory, "fit_ctx", "Piso do Fit (tokens)", 6, 0)
        self._field(memory, "ctx_checkpoints", "Checkpoints de contexto", 7, 0)
        self._field(memory, "checkpoint_min_step", "Passo mín. checkpoint", 7, 1)

        cpu = self._section(body, "CPU E BATCH", 0, 1)
        self._field(cpu, "threads", "Threads geração", 0, 0)
        self._field(cpu, "threads_batch", "Threads prompt", 0, 1)
        self._field(cpu, "batch", "Batch", 1, 0)
        self._field(cpu, "ubatch", "Micro-batch", 1, 1)
        self._field(cpu, "n_cpu_moe", "Camadas MoE CPU", 2, 0)
        self._field(cpu, "n_cpu_ffn", "Camadas FFN CPU", 2, 1)
        self._field(cpu, "load_mode", "Modo de carga", 3, 0, ["none", "mmap", "mlock", "mmap+mlock", "dio"])
        self._field(cpu, "tensor_read_lazy", "Lazy tensor loading", 3, 1, ["auto", "on", "off"])
        self._field(cpu, "numa", "NUMA", 4, 0, ["none", "distribute", "isolate", "numactl"])
        self._field(cpu, "device", "Device", 4, 1)

        sampling = self._section(body, "QUALIDADE E SAMPLING", 1, 0)
        self._field(sampling, "temp", "Temperatura", 0, 0)
        self._field(sampling, "top_k", "Top K", 0, 1)
        self._field(sampling, "top_p", "Top P", 1, 0)
        self._field(sampling, "min_p", "Min P", 1, 1)
        self._field(sampling, "repeat_penalty", "Repeat penalty", 2, 0)
        self._field(sampling, "seed", "Seed (-1 diário)", 2, 1)
        self._field(sampling, "reasoning", "Reasoning", 3, 0, ["auto", "on", "off"])
        self._field(sampling, "reasoning_budget", "Reasoning budget", 3, 1)
        self._field(sampling, "reasoning_preserve", "Preservar reasoning", 4, 0, ["auto", "y", "n"])
        self._field(sampling, "backend_sampling", "Backend sampling", 4, 1, ["auto", "y", "n"])

        advanced = self._section(body, "MULTIMODAL E EXECUÇÃO", 1, 1)
        self._switch(advanced, "omni", "Visão / multimodal", 0, 0)
        self._field(advanced, "mmproj_offload", "MMProj offload", 0, 1, ["y", "n"])
        self._field(advanced, "spec_type", "Speculative", 1, 0, list(SPECULATIVE_TYPES))
        self._field(advanced, "spec_draft_n_max", "Draft N máximo", 1, 1)
        self._switch(advanced, "cont_batching", "Continuous batching", 2, 0)
        self._switch(advanced, "cache_prompt", "Prompt cache", 2, 1)
        self._switch(advanced, "cache_idle_slots", "Cache idle slots", 3, 0)
        self._switch(advanced, "offline", "Offline llama.cpp", 3, 1)
        self._field(advanced, "host", "Host", 4, 0)
        self._field(advanced, "port", "Porta pública", 4, 1)
        self._switch(advanced, "agentic", "Agente nativo (--agent)", 5, 0)
        self._field(advanced, "tools", "Ferramentas sem --agent", 5, 1, ["all", "readonly", "none"])

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=2, pady=12, sticky="ew")
        ctk.CTkButton(
            actions, text="RECALCULAR PARA O HARDWARE", height=42,
            fg_color=GREEN_DARK, hover_color=GREEN_HOVER,
            command=lambda: self._schedule_recalc("cache_k", immediate=True),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            actions, text="RESTAURAR PERFIL AUTOMÁTICO", height=42,
            fg_color=CARD_ALT, hover_color=BORDER, command=self._restore_profile,
        ).pack(side="left", padx=8)
        self.reasons_label = ctk.CTkLabel(
            body, text="Selecione um modelo para ver as decisões do otimizador.",
            text_color=MUTED, justify="left", anchor="w", wraplength=1000, font=MONO,
        )
        self.reasons_label.grid(row=3, column=0, columnspan=2, padx=8, pady=(0, 18), sticky="ew")

    def _section(self, parent: Any, title: str, row: int, column: int) -> ctk.CTkFrame:
        card = self._card(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(card, text=title, text_color=GREEN, font=MONO).grid(
            row=0, column=0, columnspan=2, padx=16, pady=(14, 6), sticky="w"
        )
        return card

    def _var(self, key: str) -> ctk.StringVar:
        if key not in self.vars:
            self.vars[key] = ctk.StringVar(value="")
        return self.vars[key]

    def _field(self, parent: Any, key: str, label: str, row: int, column: int, choices: list[str] | None = None) -> None:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row + 1, column=column, padx=12, pady=7, sticky="ew")
        box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(box, text=label, text_color=MUTED, font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w")
        variable = self._var(key)
        if choices:
            widget = ctk.CTkOptionMenu(
                box, variable=variable, values=choices, height=32, fg_color=BG,
                button_color=GREEN_DARK, button_hover_color=GREEN_HOVER,
                dropdown_fg_color=CARD_ALT, text_color=TEXT,
                command=lambda _value, field=key: self._field_changed(field),
            )
        else:
            widget = ctk.CTkEntry(
                box, textvariable=variable, height=32, fg_color=BG,
                border_color=BORDER, text_color=TEXT,
            )
            widget.bind("<Return>", lambda _event, field=key: self._field_changed(field))
            widget.bind("<FocusOut>", lambda _event, field=key: self._field_changed(field))
        widget.grid(row=1, column=0, pady=(3, 0), sticky="ew")
        self.fields[key] = widget

    def _switch(self, parent: Any, key: str, label: str, row: int, column: int) -> None:
        variable = self._var(key)
        switch = ctk.CTkSwitch(
            parent, text=label, variable=variable, onvalue="y", offvalue="n",
            progress_color=GREEN_DARK, button_hover_color=GREEN,
            text_color=TEXT, font=ctk.CTkFont(size=11),
            command=lambda field=key: self._field_changed(field),
        )
        switch.grid(row=row + 1, column=column, padx=12, pady=11, sticky="w")
        self.fields[key] = switch

    # --------------------------------------------------------------- processo
    def _build_process_page(self) -> None:
        page = self._new_page("process")
        page.grid_columnconfigure(0, weight=2)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._title(head, "Processo", "Comando real, prontidão /health + /props e fluxo do llama-server")

        controls = self._card(page)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        controls.grid_columnconfigure(3, weight=1)
        self.start_button = ctk.CTkButton(
            controls, text="▶  INICIAR SERVIDOR", height=42, fg_color=GREEN,
            text_color="#031008", hover_color="#79ffae", command=self._start_server,
        )
        self.start_button.grid(row=0, column=0, padx=(14, 7), pady=14)
        self.stop_button = ctk.CTkButton(
            controls, text="■  ENCERRAR", height=42, fg_color="#58232b",
            hover_color="#8a3040", text_color="#ffb1ba", command=self._stop_server,
        )
        self.stop_button.grid(row=0, column=1, padx=7, pady=14)
        ctk.CTkButton(
            controls, text="PREVIEW CMD", height=42, fg_color=CARD_ALT,
            hover_color=BORDER, command=self._preview_command,
        ).grid(row=0, column=2, padx=7, pady=14)
        ctk.CTkButton(
            controls, text="LIMPAR LOG", width=110, height=32, fg_color="transparent",
            border_width=1, border_color=BORDER, hover_color=CARD_ALT,
            command=lambda: self._replace_text(self.log_box, ""),
        ).grid(row=0, column=4, padx=14, pady=14)

        self.log_box = ctk.CTkTextbox(
            page, fg_color="#030806", border_width=1, border_color=BORDER,
            text_color="#b9d9c3", font=MONO, wrap="none",
        )
        self.log_box.grid(row=2, column=0, padx=(0, 8), sticky="nsew")

        panel = self._card(page)
        panel.grid(row=2, column=1, padx=(8, 0), sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        self.process_state_label = ctk.CTkLabel(
            panel, text="◇  IDLE", text_color=GREEN, font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.process_state_label.grid(row=0, column=0, padx=18, pady=(20, 8), sticky="w")
        self.process_info_label = ctk.CTkLabel(
            panel, text="Nenhum processo ativo.", text_color=MUTED,
            justify="left", anchor="nw", wraplength=310, font=MONO,
        )
        self.process_info_label.grid(row=1, column=0, padx=18, pady=(0, 14), sticky="ew")
        ctk.CTkFrame(panel, height=1, fg_color=BORDER).grid(row=2, column=0, padx=16, sticky="ew")

        self.mcp_var = ctk.StringVar(value="n")
        self.agent_var = ctk.StringVar(value="n")
        self.snn_var = ctk.StringVar(value="n")
        self.mcp_switch = ctk.CTkSwitch(
            panel, text="MCP Crono Matrix", variable=self.mcp_var, onvalue="y", offvalue="n",
            progress_color=GREEN_DARK, command=self._mcp_changed,
        )
        self.mcp_switch.grid(row=3, column=0, padx=18, pady=(16, 8), sticky="w")
        self.agent_switch = ctk.CTkSwitch(
            panel, text="Agente local universal", variable=self.agent_var,
            onvalue="y", offvalue="n", progress_color=GREEN_DARK,
            command=self._agent_changed,
        )
        self.agent_switch.grid(row=4, column=0, padx=18, pady=8, sticky="w")
        self.snn_switch = ctk.CTkSwitch(
            panel, text="Núcleo SNN", variable=self.snn_var, onvalue="y", offvalue="n",
            progress_color=GREEN_DARK, command=self._snn_changed,
        )
        self.snn_switch.grid(row=5, column=0, padx=18, pady=8, sticky="w")
        self.agent_info_label = ctk.CTkLabel(
            panel, text="", text_color=MUTED, justify="left", anchor="nw",
            wraplength=310, font=ctk.CTkFont(family="DejaVu Sans Mono", size=10),
        )
        self.agent_info_label.grid(row=6, column=0, padx=18, pady=12, sticky="ew")

        ctk.CTkFrame(panel, height=1, fg_color=BORDER).grid(row=7, column=0, padx=16, sticky="ew")
        self.snn_status_label = ctk.CTkLabel(
            panel, text="SNN: aguardando", text_color=MUTED, font=ctk.CTkFont(size=11),
        )
        self.snn_status_label.grid(row=8, column=0, padx=18, pady=(12, 4), sticky="w")
        self.snn_confidence_label = ctk.CTkLabel(
            panel, text="", text_color=CYAN, font=ctk.CTkFont(size=10),
        )
        self.snn_confidence_label.grid(row=9, column=0, padx=18, pady=(0, 12), sticky="w")
        if not BUNDLED_MCP_AVAILABLE:
            # A edição core-only não distribui o MCP/SNN externo. Os widgets
            # permanecem instanciados porque o renderizador é compartilhado,
            # mas recursos ausentes não aparecem como controles utilizáveis.
            self.mcp_var.set("n")
            self.snn_var.set("n")
            self.mcp_switch.grid_remove()
            self.snn_switch.grid_remove()
            self.snn_status_label.grid_remove()
            self.snn_confidence_label.grid_remove()

    # ---------------------------------------------------------------- sistema
    def _build_system_page(self) -> None:
        page = self._new_page("system")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        self._title(head, "Sistema", "Telemetria usada pelo cálculo; atualize antes de trocar carga ou perfil")

        body = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        self.system_body = body
        body.grid_columnconfigure((0, 1), weight=1)
        body.grid_rowconfigure((0, 1, 2), weight=1)
        self.cpu_card = self._metric_card(body, "CPU E MEMÓRIA", 0, 0)
        self.gpu_card = self._metric_card(body, "GPU CUDA", 0, 1)
        self.storage_card = self._metric_card(body, "ARMAZENAMENTO E BUILD", 1, 0)
        self.optimizer_card = self._metric_card(body, "ESTADO DO OTIMIZADOR", 1, 1)
        self.memory_guard_card = self._metric_card(body, "GUARDA DE MEMÓRIA C99", 2, 0)
        self.memory_guard_card2 = self._metric_card(body, "CGROUP / SCOPE", 2, 1)
        ctk.CTkButton(
            body, text="ATUALIZAR TELEMETRIA", height=40, fg_color=GREEN_DARK,
            hover_color=GREEN_HOVER, command=self._refresh_hardware,
        ).grid(row=3, column=0, padx=8, pady=12, sticky="ew")
        self.swap_apply_button = ctk.CTkButton(
            body, text="APLICAR SWAP DINÂMICO", height=40, fg_color=GREEN_DARK,
            hover_color=GREEN_HOVER, command=self._apply_dynamic_swap,
        )
        self.swap_apply_button.grid(row=3, column=1, padx=8, pady=12, sticky="ew")
        self.swap_auto_button = ctk.CTkButton(
            body, text="SWAP AUTOMÁTICO: ON", height=34, fg_color=CARD_ALT,
            border_width=1, border_color=BORDER, command=self._toggle_auto_swap,
        )
        self.swap_auto_button.grid(row=4, column=0, padx=8, pady=(0, 12), sticky="ew")
        self.swap_remove_button = ctk.CTkButton(
            body, text="REMOVER SWAP NVMe", height=34, fg_color="#4e1e28",
            hover_color="#7a2a38", command=self._remove_nvme_swap,
        )
        self.swap_remove_button.grid(row=4, column=1, padx=8, pady=(0, 12), sticky="ew")

    def _metric_card(self, parent: Any, title: str, row: int, column: int) -> ctk.CTkLabel:
        card = self._card(parent)
        card.grid(row=row, column=column, padx=8, pady=8, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text=title, text_color=GREEN, font=MONO).grid(
            row=0, column=0, padx=18, pady=(15, 8), sticky="w"
        )
        value = ctk.CTkLabel(
            card, text="Aguardando…", text_color=TEXT, justify="left",
            anchor="nw", wraplength=490, font=MONO,
        )
        value.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="nsew")
        return value

    # ----------------------------------------------------------- avaliação
    def _build_eval_page(self) -> None:
        page = self._new_page("eval")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        self._title(head, "Alpha Eval Observatory", "Suite de 23 eixos — execute benchmarks e visualize resultados")

        body = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        runner_card = self._card(body)
        runner_card.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        runner_card.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(runner_card, text="EXECUTAR AVALIAÇÃO", text_color=GREEN, font=MONO).grid(
            row=0, column=0, columnspan=4, padx=16, pady=(14, 8), sticky="w"
        )

        eval_labels = [
            ("Eixos", "axes", "1,2,3", True),
            ("Repetições", "repeats", "1", False),
            ("Seed", "seed", "0", False),
            ("Escala", "scale", ["auto", "small", "medium", "large", "xlarge"], True),
            ("Token Qwen", "mode", ["auto", "think", "nothink"], True),
            ("Esforço reasoning", "reasoning_effort", ["default", "off", "low", "medium", "high", "max"], True),
            ("Budget thinking", "reasoning_budget", "auto", False),
            ("URL API", "eval_url", "http://127.0.0.1:8080/v1/chat/completions", False),
        ]
        for i, (label, key, default, is_choice) in enumerate(eval_labels):
            row = i // 2
            col = i % 2
            ctk.CTkLabel(runner_card, text=label, text_color=MUTED, font=ctk.CTkFont(size=11)).grid(
                row=row + 1, column=col, padx=8, pady=5, sticky="w"
            )
            var = self._var(f"eval_{key}")
            var.set(str(default))
            if is_choice:
                widget = ctk.CTkOptionMenu(
                    runner_card, variable=var, values=default if isinstance(default, list) else [default, default],
                    height=32, fg_color=BG, button_color=GREEN_DARK, button_hover_color=GREEN_HOVER,
                    dropdown_fg_color=CARD_ALT, text_color=TEXT,
                )
            else:
                widget = ctk.CTkEntry(
                    runner_card, textvariable=var, height=32, fg_color=BG,
                    border_color=BORDER, text_color=TEXT,
                )
            widget.grid(row=row + 1, column=col + (0 if col == 0 else 1), padx=8, pady=5, sticky="ew")

        adv_card = self._card(body)
        adv_card.grid(row=1, column=0, padx=8, pady=8, sticky="ew")
        adv_card.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(adv_card, text="CONFIGURAÇÃO DE INFERÊNCIA", text_color=CYAN, font=MONO).grid(
            row=0, column=0, columnspan=4, padx=16, pady=(14, 8), sticky="w"
        )
        adv_labels = [
            ("Sampling", "sampling", ["server", "fixed"], True),
            ("Temperatura", "temperature", "0.6", False),
            ("Top K", "top_k", "20", False),
            ("Top P", "top_p", "0.95", False),
            ("Min P", "min_p", "0.05", False),
            ("Repeat penalty", "repeat_penalty", "1.0", False),
            ("Máx. saída", "max_tokens", "16384", False),
            ("Timeout (s)", "timeout", "300", False),
            ("Ctx longo", "xctx_scale", "1.0", False),
            ("SO (Eixo 8)", "os_filter", ["all", "linux", "win"], True),
        ]
        for i, (label, key, default, is_choice) in enumerate(adv_labels):
            row = i // 2
            col = (i % 2) * 2
            ctk.CTkLabel(adv_card, text=label, text_color=MUTED, font=ctk.CTkFont(size=11)).grid(
                row=row + 1, column=col, padx=8, pady=5, sticky="w"
            )
            var = self._var(f"eval_adv_{key}")
            var.set(str(default))
            if is_choice:
                widget = ctk.CTkOptionMenu(
                    adv_card, variable=var, values=default if isinstance(default, list) else [default, default],
                    height=32, fg_color=BG, button_color=GREEN_DARK, button_hover_color=GREEN_HOVER,
                    dropdown_fg_color=CARD_ALT, text_color=TEXT,
                )
            else:
                widget = ctk.CTkEntry(
                    adv_card, textvariable=var, height=32, fg_color=BG,
                    border_color=BORDER, text_color=TEXT,
                )
            widget.grid(row=row + 1, column=col + 1, padx=8, pady=5, sticky="ew")

        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=8, pady=12, sticky="ew")
        self.eval_start_btn = ctk.CTkButton(
            btn_row, text="▶  INICIAR AVALIAÇÃO", height=42, fg_color=GREEN,
            text_color="#031008", hover_color="#79ffae", command=self._start_eval,
        )
        self.eval_start_btn.pack(side="left", padx=(0, 8))
        self.eval_stop_btn = ctk.CTkButton(
            btn_row, text="■  PARAR", height=42, fg_color="#58232b",
            hover_color="#8a3040", text_color="#ffb1ba", command=self._stop_eval,
            state="disabled",
        )
        self.eval_stop_btn.pack(side="left", padx=8)
        self.eval_export_btn = ctk.CTkButton(
            btn_row, text="📦  EXPORTAR ZIP", height=42, fg_color=CARD_ALT,
            hover_color=BORDER, command=self._export_eval,
        )
        self.eval_export_btn.pack(side="left", padx=8)

        prog_card = self._card(body)
        prog_card.grid(row=3, column=0, padx=8, pady=8, sticky="ew")
        prog_card.grid_columnconfigure(0, weight=1)
        self.eval_progress_label = ctk.CTkLabel(
            prog_card, text="Aguardando início da avaliação...", text_color=TEXT,
            font=ctk.CTkFont(size=12), anchor="w",
        )
        self.eval_progress_label.grid(row=0, column=0, padx=16, pady=10, sticky="ew")
        self.eval_prog_bar = ctk.CTkProgressBar(prog_card, width=600, height=8, progress_color=GREEN)
        self.eval_prog_bar.grid(row=1, column=0, padx=16, pady=(0, 12))
        self.eval_prog_bar.set(0)

        self.eval_log_box = ctk.CTkTextbox(
            body, fg_color="#030806", border_width=1, border_color=BORDER,
            text_color="#b9d9c3", font=MONO, wrap="none",
        )
        self.eval_log_box.grid(row=4, column=0, padx=8, pady=8, sticky="nsew")
        self.eval_log_box.configure(state="disabled")

        dash_card = self._card(body)
        dash_card.grid(row=5, column=0, padx=8, pady=8, sticky="ew")
        dash_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dash_card, text="DASHBOARD DE RESULTADOS", text_color=GREEN, font=MONO).grid(
            row=0, column=0, padx=16, pady=(14, 8), sticky="w"
        )
        self.eval_dashboard_text = ctk.CTkTextbox(
            dash_card, fg_color="transparent", border_width=0, text_color=TEXT,
            font=ctk.CTkFont(size=11), wrap="word",
        )
        self.eval_dashboard_text.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")
        dash_card.grid_rowconfigure(1, weight=1)
        self.eval_dashboard_text.configure(state="disabled")

    def _start_eval(self) -> None:
        def run_eval() -> None:
            config = self._collect_eval_config()
            self.eval_runner.start_run(**config)
        self._run_task(run_eval, self._eval_started, "Iniciando avaliação...")

    def _collect_eval_config(self) -> dict:
        def ev(key: str, default: str = "") -> str:
            v = self.vars.get(f"eval_{key}")
            return v.get() if v else str(default)
        def eva(key: str, default: str = "") -> str:
            v = self.vars.get(f"eval_adv_{key}")
            return v.get() if v else str(default)
        return {
            "axes_filter": ev("axes", ""),
            "repeats": int(ev("repeats", "1") or "1"),
            "seed": int(ev("seed", "0") or "0"),
            "scale": ev("scale", "auto"),
            "mode": ev("mode", "auto"),
            "reasoning_effort": ev("reasoning_effort", "default"),
            "reasoning_budget": ev("reasoning_budget", "auto"),
            "api_url": ev("eval_url", "http://127.0.0.1:8080/v1/chat/completions"),
            "sampling": eva("sampling", "server"),
            "temperature": float(eva("temperature", "0.6") or "0.6"),
            "top_k": int(eva("top_k", "20") or "20"),
            "top_p": float(eva("top_p", "0.95") or "0.95"),
            "min_p": float(eva("min_p", "0.05") or "0.05"),
            "repeat_penalty": float(eva("repeat_penalty", "1.0") or "1.0"),
            "max_tokens": int(eva("max_tokens", "16384") or "16384"),
            "timeout": int(eva("timeout", "300") or "300"),
            "xctx_scale": float(eva("xctx_scale", "1.0") or "1.0"),
            "os_filter": eva("os_filter", "all"),
        }

    def _eval_started(self, _result: Any = None) -> None:
        self.eval_start_btn.configure(state="disabled")
        self.eval_stop_btn.configure(state="normal")
        self._set_status("Avaliação em execução...")

    def _stop_eval(self) -> None:
        self.eval_runner.stop_run()
        self.eval_start_btn.configure(state="normal")
        self.eval_stop_btn.configure(state="disabled")
        self._set_status("Avaliação parada.")

    def _export_eval(self) -> None:
        try:
            filename, content = self.eval_runner.academic_export()
            path = filedialog.asksaveasfilename(
                defaultextension=".zip",
                filetypes=[("ZIP files", "*.zip")],
                initialfilename=filename,
                parent=self,
            )
            if path:
                Path(path).write_bytes(content)
                self._set_status(f"Exportado: {path}")
        except ValueError as exc:
            messagebox.showerror("Exportar", str(exc), parent=self)

    def _render_eval_snapshot(self, snapshot: dict) -> None:
        state = snapshot.get("state", "idle")
        progress = snapshot.get("progress", {})
        if state in ("running", "stopping"):
            total = progress.get("total", 0)
            current = progress.get("current", 0)
            passed = progress.get("passed", 0)
            failed = progress.get("failed", 0)
            skipped = progress.get("skipped", 0)
            axis = progress.get("axis", "")
            test = progress.get("test", "")
            pct = (current / total * 100) if total > 0 else 0
            self.eval_progress_label.configure(
                text=f"EIXO {axis}  ·  {test}  ·  {current}/{total}  ·  ✓{passed}  ✗{failed}  ⏭{skipped}  ({pct:.0f}%)"
            )
            self.eval_prog_bar.set(pct / 100)
            self.eval_start_btn.configure(state="disabled")
            self.eval_stop_btn.configure(state="normal")
        elif state in ("done", "error", "stopped", "skipped"):
            self.eval_progress_label.configure(
                text=f"Estado: {state.upper()}  ·  ✓{progress.get('passed',0)}  ✗{progress.get('failed',0)}  ⏭{progress.get('skipped',0)}"
            )
            self.eval_prog_bar.set(1.0 if state == "done" else 0)
            self.eval_start_btn.configure(state="normal")
            self.eval_stop_btn.configure(state="disabled")
        else:
            self.eval_progress_label.configure(text="Aguardando início da avaliação...")
            self.eval_prog_bar.set(0)

    def _render_eval_logs(self, logs: list) -> None:
        if not logs:
            return
        self.eval_log_box.configure(state="normal")
        for item in logs:
            level = item.get("level", "")
            line = item.get("line", "")
            color = "#ff6474" if level in ("error", "fail") else (
                "#efc56b" if level in ("warn",) else (
                    "#42f58d" if level in ("pass", "success") else "#b9d9c3"
                )
            )
            self.eval_log_box.insert("end", line + "\n")
        self.eval_log_box.see("end")
        self.eval_log_box.configure(state="disabled")

    def _render_eval_dashboard(self, dashboard_data: dict | None) -> None:
        self.eval_dashboard_text.configure(state="normal")
        self.eval_dashboard_text.delete("1.0", "end")
        if not dashboard_data or not dashboard_data.get("checkpoints"):
            self.eval_dashboard_text.insert("end", "Nenhuma avaliação concluída ainda.\n\nExecute a suite de avaliação para ver os resultados aqui.")
        else:
            cps = dashboard_data.get("checkpoints", [])
            self.eval_dashboard_text.insert("end", f"{len(cps)} execução(ões) registrada(s)\n\n")
            for cp in cps[:5]:
                summary = cp.get("summary", {})
                meta = cp.get("baseModel", "modelo")
                fp = cp.get("configFingerprint", "")[:8]
                label = cp.get("variantLabel", "")
                mean = summary.get("meanScore", 0)
                pass_rate = summary.get("passRate", 0)
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                self.eval_dashboard_text.insert("end", (
                    f"  {meta}  [{fp}]\n"
                    f"    Score: {mean:.2f}/10  ·  Aprovação: {pass_rate*100:.0f}%  ·  "
                    f"✓{passed}  ✗{failed}\n"
                    f"    {label}\n\n"
                ))
            self.eval_dashboard_text.insert("end", "...")
        self.eval_dashboard_text.configure(state="disabled")

    # -------------------------------------------------------------- radar
    def _build_radar_page(self) -> None:
        page = self._new_page("radar")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        self._title(
            head, "Radar de Lançamentos · Online",
            "Recurso opcional: a rede só é acessada ao atualizar ou permitir consultas",
        )

        body = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        radar_card = self._card(body)
        radar_card.grid(row=0, column=0, padx=8, pady=6, sticky="nsew")
        radar_card.grid_rowconfigure(2, weight=1)
        radar_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(radar_card, text="HUGGING FACE · REDE OPCIONAL", text_color=GREEN, font=MONO).grid(
            row=0, column=0, columnspan=2, padx=14, pady=(10, 4), sticky="w"
        )
        self.radar_status_label = ctk.CTkLabel(
            radar_card, text="Nenhuma consulta de rede nesta sessão.", text_color=MUTED,
            font=ctk.CTkFont(size=10), anchor="w",
        )
        self.radar_status_label.grid(row=1, column=0, padx=14, pady=2, sticky="w")
        self.radar_unread_badge = ctk.CTkLabel(
            radar_card, text="", text_color=AMBER, font=ctk.CTkFont(size=10, weight="bold"),
        )
        self.radar_unread_badge.grid(row=1, column=1, padx=8, pady=2, sticky="e")

        self.radar_list = ctk.CTkScrollableFrame(radar_card, fg_color="transparent", corner_radius=0)
        self.radar_list.grid(row=2, column=0, columnspan=2, padx=8, pady=4, sticky="nsew")
        self.radar_list.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.grid(row=1, column=0, padx=8, pady=4, sticky="ew")
        ctk.CTkButton(
            btn_row, text="↻  ATUALIZAR", height=32, fg_color=GREEN_DARK,
            hover_color=GREEN_HOVER, command=self._refresh_radar,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="✓  VISTO", height=32, fg_color=CARD_ALT,
            hover_color=BORDER, command=self._mark_radar_read,
        ).pack(side="left", padx=6)

        detail_card = self._card(body)
        detail_card.grid(row=2, column=0, padx=8, pady=6, sticky="ew")
        detail_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            detail_card, text="SELEÇÃO E DOWNLOAD", text_color=CYAN, font=MONO,
        ).grid(row=0, column=0, padx=14, pady=(10, 2), sticky="w")
        self.hf_detail_title = ctk.CTkLabel(
            detail_card,
            text="Escolha DETALHES em um lançamento para listar seus GGUFs.",
            text_color=TEXT, font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w", justify="left", wraplength=1050,
        )
        self.hf_detail_title.grid(row=1, column=0, padx=14, pady=(2, 0), sticky="ew")
        self.hf_detail_meta = ctk.CTkLabel(
            detail_card, text="Nenhum download é iniciado sem sua confirmação.",
            text_color=MUTED, font=ctk.CTkFont(size=10), anchor="w", justify="left",
        )
        self.hf_detail_meta.grid(row=2, column=0, padx=14, pady=(2, 5), sticky="ew")
        self.hf_file_var = ctk.StringVar(value="")
        self.hf_file_list = ctk.CTkScrollableFrame(
            detail_card, height=155, fg_color=BG, corner_radius=8,
            border_width=1, border_color=BORDER,
        )
        self.hf_file_list.grid(row=3, column=0, padx=14, pady=5, sticky="ew")
        self.hf_file_list.grid_columnconfigure(0, weight=1)
        self.hf_file_empty = ctk.CTkLabel(
            self.hf_file_list, text="Nenhum repositório selecionado.",
            text_color=MUTED, font=ctk.CTkFont(size=10),
        )
        self.hf_file_empty.grid(row=0, column=0, padx=10, pady=18, sticky="w")

        download_row = ctk.CTkFrame(detail_card, fg_color="transparent")
        download_row.grid(row=4, column=0, padx=14, pady=(5, 2), sticky="ew")
        download_row.grid_columnconfigure(2, weight=1)
        self.hf_download_button = ctk.CTkButton(
            download_row, text="↓  BAIXAR GGUF SELECIONADO", height=34,
            fg_color=GREEN_DARK, hover_color=GREEN_HOVER,
            state="disabled", command=self._start_hf_download,
        )
        self.hf_download_button.grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.hf_cancel_button = ctk.CTkButton(
            download_row, text="CANCELAR", width=100, height=34,
            fg_color="#58232b", hover_color="#8a3040", text_color="#ffb1ba",
            state="disabled", command=self._cancel_hf_download,
        )
        self.hf_cancel_button.grid(row=0, column=1, padx=6, sticky="w")
        self.hf_download_status = ctk.CTkLabel(
            download_row, text="Aguardando seleção", text_color=MUTED,
            font=ctk.CTkFont(size=10), anchor="e",
        )
        self.hf_download_status.grid(row=0, column=2, padx=(8, 0), sticky="e")
        self.hf_download_progress = ctk.CTkProgressBar(
            detail_card, height=7, progress_color=GREEN, fg_color=CARD_ALT,
        )
        self.hf_download_progress.grid(row=5, column=0, padx=14, pady=(3, 12), sticky="ew")
        self.hf_download_progress.set(0)

        settings_card = self._card(body)
        settings_card.grid(row=3, column=0, padx=8, pady=6, sticky="ew")
        settings_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(settings_card, text="PREFERÊNCIAS DO RADAR", text_color=CYAN, font=MONO).grid(
            row=0, column=0, columnspan=2, padx=14, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(settings_card, text="Famílias observadas:", text_color=MUTED, font=ctk.CTkFont(size=10)).grid(
            row=1, column=0, padx=14, pady=3, sticky="w"
        )
        self.radar_watchlist_var = ctk.StringVar(value="qwen,gemma,glm,deepseek,mistral,llama,nemotron,gpt-oss")
        ctk.CTkEntry(
            settings_card, textvariable=self.radar_watchlist_var, height=28, fg_color=BG,
            border_color=BORDER, text_color=TEXT,
        ).grid(row=1, column=1, padx=8, pady=3, sticky="ew")
        self.radar_enabled_var = ctk.StringVar(value="y")
        ctk.CTkSwitch(
            settings_card, text="Permitir consultas de lançamentos",
            variable=self.radar_enabled_var, onvalue="y", offvalue="n",
            progress_color=GREEN_DARK, command=self._radar_prefs_changed,
        ).grid(row=2, column=0, columnspan=2, padx=14, pady=4, sticky="w")
        ctk.CTkButton(
            settings_card, text="SALVAR", height=32, fg_color=GREEN_DARK,
            hover_color=GREEN_HOVER, command=self._save_radar_prefs,
        ).grid(row=3, column=0, columnspan=2, padx=14, pady=6, sticky="w")

    def _refresh_radar(self) -> None:
        self._run_task(
            lambda: self.backend.refresh_hf_radar(force=True),
            self._render_radar_snapshot,
            "Atualizando radar...",
        )

    def _mark_radar_read(self) -> None:
        self._run_task(
            lambda: self.backend.mark_hf_radar_read(),
            self._render_radar_snapshot,
            "Marcando como visto...",
        )

    def _save_radar_prefs(self) -> None:
        watchlist = self.radar_watchlist_var.get()
        enabled = self.radar_enabled_var.get() == "y"
        self._run_task(
            lambda: self.backend.set_hf_radar_preferences(watchlist, enabled),
            self._render_radar_snapshot,
            "Salvando preferências...",
        )

    def _radar_prefs_changed(self) -> None:
        self._save_radar_prefs()

    def _open_radar_model(self, repo_id: str) -> None:
        if not repo_id or "/" not in repo_id:
            self._set_status("Repositório Hugging Face inválido.", error=True)
            return
        self._radar_selected_repo = repo_id
        self.hf_file_var.set("")
        self.hf_detail_title.configure(text=repo_id)
        self.hf_detail_meta.configure(text="Consultando arquivos GGUF e metadados…", text_color=MUTED)
        self.hf_download_button.configure(state="disabled")
        self._run_task(
            lambda: self.backend.hf_details(repo_id),
            self._render_hf_detail,
            f"Consultando {repo_id}…",
        )

    def _render_hf_detail(self, detail: dict) -> None:
        for child in self.hf_file_list.winfo_children():
            child.destroy()
        repo_id = str(detail.get("repo_id") or self._radar_selected_repo)
        original = str(detail.get("original") or repo_id)
        self._radar_selected_repo = repo_id
        self.hf_detail_title.configure(text=repo_id)
        meta = detail.get("meta") if isinstance(detail.get("meta"), dict) else {}
        arch = str(meta.get("architecture") or "arquitetura não informada").upper()
        context = compact_int(meta.get("context_length") or 0)
        params = compact_int(meta.get("total") or 0)
        conversion = f"Conversão GGUF de {original}  ·  " if original != repo_id else ""
        self.hf_detail_meta.configure(
            text=f"{conversion}{arch}  ·  contexto {context}  ·  parâmetros {params}",
            text_color=MUTED,
        )
        files = detail.get("files") if isinstance(detail.get("files"), list) else []
        self.hf_file_var.set("")
        for row, item in enumerate(files):
            filename = str(item.get("name") or "")
            label = str(item.get("label") or Path(filename).stem)
            parts = max(int(item.get("parts") or 1), 1)
            size = format_bytes(item.get("size") or 0) if item.get("size") else "tamanho N/D"
            suffix = f"  ·  {parts} partes" if parts > 1 else ""
            ctk.CTkRadioButton(
                self.hf_file_list,
                text=f"{label}  ·  {size}{suffix}",
                variable=self.hf_file_var, value=filename,
                text_color=TEXT, fg_color=GREEN_DARK, hover_color=GREEN,
                font=ctk.CTkFont(family="DejaVu Sans Mono", size=10),
                command=self._hf_file_selected,
            ).grid(row=row, column=0, padx=10, pady=6, sticky="w")
        if not files:
            ctk.CTkLabel(
                self.hf_file_list,
                text="Nenhuma conversão GGUF exata foi localizada para este modelo.",
                text_color=AMBER, font=ctk.CTkFont(size=10),
            ).grid(row=0, column=0, padx=10, pady=18, sticky="w")
        self._hf_file_selected()
        self._set_status(f"{len(files)} opção(ões) GGUF em {repo_id}")

    def _hf_file_selected(self) -> None:
        state = hf_download_view(self.backend.download_snapshot())["state"]
        enabled = bool(self.hf_file_var.get()) and state not in {"running", "cancelling"}
        self.hf_download_button.configure(state="normal" if enabled else "disabled")

    def _start_hf_download(self) -> None:
        repo_id = self._radar_selected_repo
        filename = self.hf_file_var.get()
        if not repo_id or not filename:
            self._set_status("Selecione um arquivo GGUF antes de baixar.", error=True)
            return
        try:
            snapshot = self.backend.start_download(repo_id, filename)
        except Exception as exc:
            self._set_status(str(exc), error=True)
            messagebox.showerror("Download Hugging Face", str(exc), parent=self)
            return
        self._download_completion_seen = ""
        self._render_hf_download(snapshot)
        self._set_status(f"Baixando {filename} para {self.backend.models_dir}…")

    def _cancel_hf_download(self) -> None:
        self._render_hf_download(self.backend.cancel_download())
        self._set_status("Cancelando download…")

    def _render_hf_download(self, snapshot: dict) -> None:
        view = hf_download_view(snapshot)
        state = view["state"]
        self.hf_download_progress.set(view["progress"])
        if state in {"running", "cancelling"}:
            detail = (
                f"{view['percent']:.1f}%  ·  {format_bytes(view['downloaded'])} / "
                f"{format_bytes(view['total'])}  ·  {format_bytes(view['speed'])}/s"
            )
            self.hf_download_status.configure(text=f"{view['label']}  {detail}", text_color=CYAN)
            self.hf_download_button.configure(state="disabled")
            self.hf_cancel_button.configure(state="normal" if state == "running" else "disabled")
        elif state == "done":
            self.hf_download_status.configure(
                text=f"CONCLUÍDO  ·  {format_bytes(view['downloaded'])}", text_color=GREEN,
            )
            self.hf_cancel_button.configure(state="disabled")
            self._hf_file_selected()
            completion_key = repr(view["paths"])
            if completion_key and completion_key != self._download_completion_seen:
                self._download_completion_seen = completion_key
                self._render_models(self.backend.models_snapshot())
                self._set_status("Download verificado e modelo adicionado à biblioteca local.")
        elif state in {"error", "cancelled"}:
            message = view["error"] or view["label"]
            self.hf_download_status.configure(text=message, text_color=RED if state == "error" else AMBER)
            self.hf_cancel_button.configure(state="disabled")
            self._hf_file_selected()
        else:
            self.hf_download_status.configure(text=view["label"], text_color=MUTED)
            self.hf_cancel_button.configure(state="disabled")
            self._hf_file_selected()

    def _render_radar_snapshot(self, radar: dict) -> None:
        last_refresh = radar.get("last_refresh", "")
        unread_count = radar.get("unread_count", 0)
        enabled = radar.get("enabled", True)
        initialized = radar.get("initialized", False)
        items = radar.get("items", [])
        error = radar.get("error", "")

        if last_refresh:
            ts = last_refresh[:16].replace("T", " ")
            self.radar_status_label.configure(text=f"Sincronizado em {ts} UTC")
        elif not initialized:
            self.radar_status_label.configure(text="Montando linha de base...")
        else:
            self.radar_status_label.configure(text="Aguardando primeira sincronização...")

        if unread_count > 0:
            self.radar_unread_badge.configure(text=f"◆ {unread_count} NOVO{'S' if unread_count > 1 else ''}")
        else:
            self.radar_unread_badge.configure(text="OK")

        for child in self.radar_list.winfo_children():
            child.destroy()
        for i, item in enumerate(items[:20]):
            repo_id = item.get("id", "")
            official = item.get("official", False)
            trusted = item.get("trusted", False)
            family = item.get("family", "").upper()
            capabilities = item.get("capabilities", [])[:3]
            unread = item.get("unread", False)
            event = item.get("event", "")

            frame = ctk.CTkFrame(
                self.radar_list,
                fg_color=CARD_ALT if not unread else CARD,
                corner_radius=8, border_width=1,
                border_color=GREEN_DARK if unread else BORDER,
            )
            frame.grid(row=i, column=0, padx=4, pady=3, sticky="ew")
            frame.grid_columnconfigure(1, weight=1)

            signal_char = "◆" if unread else "◇"
            signal_color = GREEN if unread else MUTED
            ctk.CTkLabel(
                frame, text=signal_char, text_color=signal_color,
                font=ctk.CTkFont(size=12),
            ).grid(row=0, column=0, padx=8, pady=6, sticky="w")

            info_text = f"{repo_id}"
            trust = "OFICIAL" if official else ("VERIFICADO" if trusted else family)
            caps = "  ".join(f"<{c}>" for c in capabilities)
            info_text += f"  [{trust}]  {caps}"
            if unread and event:
                info_text += f"  → {event}"

            ctk.CTkLabel(
                frame, text=info_text, text_color=TEXT if unread else MUTED,
                font=ctk.CTkFont(size=11), anchor="w",
            ).grid(row=0, column=1, padx=8, pady=6, sticky="ew")
            ctk.CTkButton(
                frame, text="DETALHES  →", width=100, height=28,
                fg_color=GREEN_DARK if unread else CARD_ALT,
                hover_color=GREEN_HOVER, text_color=TEXT,
                command=lambda selected=repo_id: self._open_radar_model(selected),
            ).grid(row=0, column=2, padx=8, pady=5, sticky="e")

        if error:
            ctk.CTkLabel(
                self.radar_list, text=f"Leitura parcial: {error}", text_color=RED,
                font=ctk.CTkFont(size=10),
            ).grid(row=len(items), column=0, padx=8, pady=4, sticky="w")

        self.radar_watchlist_var.set(radar.get("watchlist", self.radar_watchlist_var.get()))
        self.radar_enabled_var.set("y" if enabled else "n")

    # ---------------------------------------------------------- async core
    def _run_task(self, action: Callable[[], Any], done: Callable[[Any], None] | None = None, status: str = "") -> None:
        self._busy += 1
        self.busy_bar.configure(mode="indeterminate")
        self.busy_bar.start()
        if status:
            self._set_status(status)

        def worker() -> None:
            try:
                result = action()
                self.async_events.put((True, result, done))
            except Exception as exc:  # exibido na UI, sem matar o loop Tk
                self.async_events.put((False, exc, None))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_async(self) -> None:
        if self._closing:
            return
        while True:
            try:
                ok, value, callback = self.async_events.get_nowait()
            except queue.Empty:
                break
            self._busy = max(0, self._busy - 1)
            if ok:
                if callback:
                    callback(value)
            else:
                self._set_status(str(value), error=True)
                messagebox.showerror("Crono Matrix", str(value), parent=self)
        if self._busy == 0:
            self.busy_bar.stop()
            self.busy_bar.configure(mode="determinate")
            self.busy_bar.set(0)
        self.after(80, self._drain_async)

    def _initial_load(self) -> tuple[dict, list[dict]]:
        hardware = self.backend.refresh_hardware()
        models = self.backend.scan_models()
        return hardware, models

    def _initial_ready(self, result: tuple[dict, list[dict]]) -> None:
        hardware, models = result
        self._render_hardware(hardware)
        self._render_models(models)
        if BUNDLED_MCP_AVAILABLE:
            self._load_snn_snapshot()
        self._set_status(f"Pronto · {len(models)} modelo(s) GGUF encontrado(s)")

    # ------------------------------------------------------------ model events
    def _choose_llama(self) -> None:
        path = filedialog.askdirectory(initialdir=self.llama_path_var.get(), parent=self)
        if path:
            self.llama_path_var.set(path)

    def _choose_models(self) -> None:
        path = filedialog.askdirectory(initialdir=self.models_path_var.get(), parent=self)
        if path:
            self.models_path_var.set(path)

    def _apply_paths(self) -> None:
        self._run_task(
            lambda: self.backend.configure_paths(self.llama_path_var.get(), self.models_path_var.get()),
            lambda _cfg: self._render_models(self.backend.models_snapshot()),
            "Validando caminhos e varrendo GGUF…",
        )

    def _render_models(self, models: list[dict]) -> None:
        for child in self.model_list.winfo_children():
            child.destroy()
        self.model_buttons.clear()
        if not models:
            ctk.CTkLabel(
                self.model_list, text="Nenhum GGUF principal encontrado.",
                text_color=MUTED,
            ).grid(row=0, column=0, padx=10, pady=20)
            return
        for row, model in enumerate(models):
            frame = ctk.CTkFrame(
                self.model_list, fg_color=CARD_ALT if model.get("selected") else CARD,
                corner_radius=9, border_width=1,
                border_color=GREEN_DARK if model.get("selected") else BORDER,
            )
            frame.grid(row=row, column=0, padx=4, pady=4, sticky="ew")
            frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                frame, text=model["name"], text_color=TEXT, anchor="w",
                font=ctk.CTkFont(size=12, weight="bold"),
            ).grid(row=0, column=0, padx=12, pady=(9, 1), sticky="ew")
            ctk.CTkLabel(
                frame, text=f"{human_size(model['size_gb'])}  ·  {model['relative']}",
                text_color=MUTED, anchor="w", font=ctk.CTkFont(size=10),
            ).grid(row=1, column=0, padx=12, pady=(0, 9), sticky="ew")
            button = ctk.CTkButton(
                frame, text="SELECIONAR", width=105, height=31,
                fg_color=GREEN_DARK, hover_color=GREEN_HOVER,
                command=lambda model_id=model["id"]: self._select_model(model_id),
            )
            button.grid(row=0, column=1, rowspan=2, padx=10, pady=9)
            self.model_buttons.append(button)

    def _select_model(self, model_id: str) -> None:
        self._run_task(lambda: self.backend.select_model(model_id), self._model_selected, "Lendo GGUF e calculando perfil…")

    def _model_selected(self, model: dict) -> None:
        self.active_model_label.configure(text=model["name"])
        self._manual_fields.clear()
        self._apply_parameter_snapshot(self.backend.parameter_snapshot())
        self._render_models(self.backend.models_snapshot())
        self._render_model_detail(model)
        self.show_page("inference")
        self._set_status(f"Perfil calculado para {model['name']}")

    def _render_model_detail(self, model: dict) -> None:
        if model.get("swa_layers"):
            layout = (
                f"{model['swa_layers']} SWA · "
                f"{model['global_layers']} global · {model['moe_layers']} MoE"
            )
        else:
            layout = f"{model['attention_layers']} atenção · {model['recurrent_layers']} SSM · {model['moe_layers']} MoE"
        text = (
            "MODELO SELECIONADO\n"
            f"{model['name']}\n\n"
            f"QUANTIZAÇÃO       {model['quant']}\n"
            f"ARQUITETURA       {model['arch']}\n"
            f"PARÂMETROS        {model['params']}\n"
            f"TAMANHO           {human_size(model['size_gb'])}\n"
            f"CONTEXTO MÁXIMO   {compact_int(model['ctx_max'])}\n"
            f"CAMADAS           {model['layers']}\n"
            f"LAYOUT            {layout}\n"
            f"EXPERTS           {model['expert_count']} / {model['expert_used_count']} ativos\n"
            f"MMProj            {model['mmproj'] or 'não detectado'}\n"
            f"MTP               {'sim' if model['has_mtp'] else 'não'}\n\n"
            "O perfil abaixo é calculado com a VRAM e RAM livres no momento da seleção."
        )
        self._replace_text(self.model_detail, text)

    # ---------------------------------------------------------- param events
    def _collect_form(self) -> dict[str, str]:
        raw = {key: variable.get() for key, variable in self.vars.items()}
        raw["mcp_native"] = self.mcp_var.get()
        raw["agent_compat"] = "y"
        raw["agent_global"] = self.agent_var.get()
        return raw

    def _field_changed(self, key: str) -> None:
        if self._updating_form or not self.backend.model_snapshot():
            return
        self._manual_fields.add(key)
        if key in self.MEMORY_FIELDS:
            self._schedule_recalc(key)

    def _schedule_recalc(self, key: str, immediate: bool = False) -> None:
        if self._recalc_job:
            self.after_cancel(self._recalc_job)
            self._recalc_job = None
        delay = 0 if immediate else 550
        self._recalc_job = self.after(delay, lambda: self._recalculate(key))

    def _recalculate(self, key: str) -> None:
        self._recalc_job = None
        raw = self._collect_form()
        raw["recalculate_field"] = key
        self._run_task(lambda: self.backend.recalculate_memory(raw), self._apply_parameter_snapshot, "Recalculando VRAM e contexto…")

    def _restore_profile(self) -> None:
        model = self.backend.model_snapshot()
        if not model:
            self._set_status("Selecione um modelo primeiro.", error=True)
            return
        def restore() -> dict:
            self.backend.refresh_hardware()
            return self.backend.restore_optimal_profile()

        def done(snapshot: dict) -> None:
            self._manual_fields.clear()
            self._apply_parameter_snapshot(snapshot)
            self._set_status("Perfil automático restaurado com a telemetria atual.")

        self._run_task(restore, done, "Atualizando hardware e restaurando perfil…")

    def _apply_parameter_snapshot(self, snapshot: dict) -> None:
        values = snapshot.get("values", {})
        planning = snapshot.get("planning_hardware", {})
        try:
            self._profile_vram_free_mb = int(
                planning.get("gpu_vram_free_mb", self._profile_vram_free_mb)
            )
        except (TypeError, ValueError):
            pass
        self._updating_form = True
        try:
            for key, value in values.items():
                if key in self.vars:
                    if isinstance(value, bool):
                        value = "y" if value else "n"
                    self.vars[key].set(str(value))
            self.mcp_var.set(str(values.get("mcp_native", "n")))
            self.agent_var.set(str(values.get("agent_global", "n")))
        finally:
            self._updating_form = False
        reasons = snapshot.get("reasons", [])
        focus = {"Contexto", "Fit efetivo", "GPU layers", "CPU MoE", "Cache KV", "Threads", "Batch", "Speculative", "Omni/visao", "Autotune"}
        lines = [f"{name}: {reason}" for name, reason in reasons if name in focus]
        self.reasons_label.configure(text="\n".join(lines) if lines else "Sem diagnóstico disponível.")
        self.optimizer_card.configure(text="\n".join(lines) if lines else "Selecione um modelo.")

    # --------------------------------------------------------- process events
    def _preview_command(self) -> None:
        def done(result: tuple[dict, list[str], str]) -> None:
            self._replace_text(self.log_box, result[2] + "\n")
            self.show_page("process")
            self._set_status("Comando validado e exibido; servidor ainda não iniciado.")
        self._run_task(lambda: self.backend.preview_command(self._collect_form()), done, "Validando comando…")

    def _start_server(self) -> None:
        self.show_page("process")
        raw = self._collect_form()
        manual_fields = set(self._manual_fields)
        planned_free_mb = self._profile_vram_free_mb

        def start_with_current_hardware() -> tuple[dict | None, dict]:
            hardware = self.backend.refresh_hardware()
            current_free_mb = int(hardware.get("gpu_vram_free_mb", 0))
            action = vram_plan_action(planned_free_mb, current_free_mb)
            launch_raw = dict(raw)
            refreshed_snapshot = None

            if action == "block":
                raise ValueError(
                    "A VRAM livre caiu desde o cálculo do perfil "
                    f"({planned_free_mb} → {current_free_mb} MiB). Encerre o "
                    "outro processo GPU e restaure/recalcule o perfil antes de iniciar."
                )
            if action == "rebuild":
                manual_memory = sorted(manual_fields & self.MEMORY_PLAN_FIELDS)
                if manual_memory:
                    raise ValueError(
                        "A VRAM mudou desde o cálculo e existem ajustes manuais de "
                        "memória (" + ", ".join(manual_memory) + "). Use "
                        "RECALCULAR PARA O HARDWARE ou RESTAURAR PERFIL AUTOMÁTICO."
                    )
                refreshed_snapshot = self.backend.restore_optimal_profile()
                fresh_values = dict(refreshed_snapshot.get("values", {}))
                # Preserva sampling, rede, ferramentas e demais escolhas que
                # não dependem da quantidade de VRAM disponível.
                for key, value in raw.items():
                    if key not in self.MEMORY_DERIVED_FIELDS:
                        fresh_values[key] = value
                launch_raw = fresh_values

            process = self.backend.start_server(launch_raw)
            if refreshed_snapshot is not None:
                # ``start_server`` grava os valores efetivos em ``params``;
                # devolva esse snapshot para a UI refletir também sampling e
                # demais escolhas manuais preservadas acima.
                refreshed_snapshot = self.backend.parameter_snapshot()
            return refreshed_snapshot, process

        def started(result: tuple[dict | None, dict]) -> None:
            snapshot, _process = result
            if snapshot:
                self._manual_fields.difference_update(self.MEMORY_DERIVED_FIELDS)
                self._apply_parameter_snapshot(snapshot)
                self._set_status(
                    "VRAM mudou; perfil recalculado automaticamente e processo iniciado."
                )
            else:
                self._set_status("Processo iniciado; aguardando /health e /props…")

        self._run_task(
            start_with_current_hardware, started,
            "Atualizando telemetria e iniciando llama-server…",
        )

    def _stop_server(self) -> None:
        self._run_task(self.backend.stop_server, lambda _value: self._set_status("Servidor encerrado."), "Encerrando processo…")

    def _mcp_changed(self) -> None:
        if self.backend.is_running():
            self.mcp_var.set("y" if self.backend.process_snapshot().get("gateway_enabled") else "n")
            self._set_status("MCP só pode ser alterado antes de iniciar o servidor.", error=True)
            return
        if self.mcp_var.get() == "y" and self.agent_var.get() == "y":
            self.agent_var.set("n")

    def _agent_changed(self) -> None:
        enabled = self.agent_var.get() == "y"
        if enabled:
            self.mcp_var.set("n")
        if self.backend.is_running():
            self._run_task(lambda: self.backend.set_agent_global(enabled), lambda _v: self._set_status("Modo universal atualizado."), "Atualizando integração universal…")

    def _snn_changed(self) -> None:
        enabled = self.snn_var.get() == "y"
        self._run_task(lambda: self.backend.set_snn_enabled(enabled), lambda _v: self._set_status("Estado SNN atualizado."), "Atualizando núcleo SNN…")

    def _load_snn_snapshot(self) -> None:
        try:
            self.snn_var.set("y" if self.backend.snn_enabled() else "n")
        except Exception:
            self.snn_var.set("n")

    def _poll_runtime(self) -> None:
        if self._closing:
            return
        try:
            process = self.backend.process_snapshot()
            process_visual = dict(process)
            process_visual.pop("memory_guard", None)
            process_key = repr(process_visual)
            if process_key != self._last_process_render:
                self._last_process_render = process_key
                self._render_process(process)
            self._render_memory_guard(process.get("memory_guard"))
            logs = self.backend.logs_after(self.log_sequence)
            if logs:
                self.log_sequence = max(item["seq"] for item in logs)
                self.log_box.configure(state="normal")
                for item in logs:
                    self.log_box.insert("end", item["line"])
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except Exception as exc:
            self._set_status(f"Falha ao atualizar processo: {exc}", error=True)
        try:
            eval_snap = self.eval_runner.snapshot()
            eval_key = repr(eval_snap)
            if eval_key != self._last_eval_render:
                self._last_eval_render = eval_key
                self._render_eval_snapshot(eval_snap)
            eval_logs = self.eval_runner.logs_after(getattr(self, "_eval_log_seq", 0))
            if eval_logs:
                self._eval_log_seq = max(item["seq"] for item in eval_logs)
                self._render_eval_logs(eval_logs)
            if eval_snap.get("state") in ("done", "error", "stopped", "skipped"):
                dashboard_key = (
                    eval_snap.get("state"), eval_snap.get("results_file"),
                    eval_snap.get("dashboard_timestamp"),
                )
                if dashboard_key != self._last_eval_dashboard_render:
                    self._last_eval_dashboard_render = dashboard_key
                    dash_data, _ = self.eval_runner.get_dashboard_data()
                    self._render_eval_dashboard(dash_data)
        except Exception:
            pass
        try:
            radar_snap = self.backend.hf_radar_snapshot()
            radar_key = repr(radar_snap)
            if radar_snap.get("initialized") and radar_key != self._last_radar_render:
                self._last_radar_render = radar_key
                self._render_radar_snapshot(radar_snap)
        except Exception:
            pass
        try:
            download_snap = self.backend.download_snapshot()
            download_key = repr(download_snap)
            if download_key != self._last_download_render:
                self._last_download_render = download_key
                self._render_hf_download(download_snap)
        except Exception:
            pass
        if BUNDLED_MCP_AVAILABLE:
            try:
                snn_snap = self.backend.snn_snapshot()
                snn_key = repr(snn_snap)
                if snn_key != self._last_snn_render:
                    self._last_snn_render = snn_key
                    self._render_snn_snapshot(snn_snap)
            except Exception:
                pass
        self.after(400, self._poll_runtime)

    def _render_snn_snapshot(self, snn: dict) -> None:
        enabled = snn.get("enabled", False)
        status = snn.get("status", "unknown")
        intent = snn.get("intent", "OFF")
        confidence = snn.get("confidence", 0.0)
        message = snn.get("message", "")
        if enabled:
            color = GREEN if status in ("online",) else (AMBER if status in ("waiting", "unavailable") else RED)
            self.snn_status_label.configure(text=f"SNN: {status.upper()}  ·  {intent}", text_color=color)
            self.snn_confidence_label.configure(text=f"Confiança: {confidence:.2f}  ·  {message}")
        else:
            self.snn_status_label.configure(text="SNN: desativado", text_color=MUTED)
            self.snn_confidence_label.configure(text="")

    def _render_process(self, process: dict) -> None:
        state = str(process.get("state", "idle")).upper()
        ready = bool(process.get("ready"))
        running = bool(process.get("running"))
        color = GREEN if ready or state == "IDLE" else (AMBER if running else RED)
        symbol = "◆" if ready else ("◇" if state == "IDLE" else "●")
        badge = "READY" if ready else state
        self.runtime_badge.configure(text=f"{symbol}  {badge}", text_color=color)
        self.process_state_label.configure(text=f"{symbol}  {badge}", text_color=color)
        runtime = process.get("runtime_effective", {})
        memory_buffers = runtime.get("memory_buffers", {})
        kv_runtime = memory_buffers.get("kv", {})
        rs_runtime = memory_buffers.get("rs", {})
        agent = process.get("agent_compat", {})
        guard = memory_guard_view(process.get("memory_guard"))
        info = (
            f"PID: {process.get('pid') or '—'}\n"
            f"Modelo: {process.get('model') or '—'}\n"
            f"API: http://{process.get('host')}:{process.get('public_port')}\n"
            f"Contexto pedido: {compact_int(runtime.get('requested_context'))}\n"
            f"Contexto efetivo: {compact_int(runtime.get('context_window'))}\n"
            f"Slots: {runtime.get('total_slots') or '—'}\n"
            f"KV real: {str(kv_runtime.get('placement') or 'não verificado').upper()}"
            f" · GPU {kv_runtime.get('gpu_mb', 0):.1f} MiB"
            f" · CPU {kv_runtime.get('cpu_mb', 0):.1f} MiB\n"
            f"Estado RS: {str(rs_runtime.get('placement') or 'não verificado').upper()}"
            f" · GPU {rs_runtime.get('gpu_mb', 0):.1f} MiB"
            f" · CPU {rs_runtime.get('cpu_mb', 0):.1f} MiB\n"
            f"RAM disponível: {compact_int(guard['available_mb'])} MiB"
            f" · cgroup: {compact_int(guard['current_mb'])} MiB"
        )
        if BUNDLED_MCP_AVAILABLE:
            info += (
                f"\nMCP: {process.get('mcp_state')} "
                f"({process.get('mcp_tools', 0)} tools)"
            )
        if process.get("error"):
            info += f"\n\nERRO: {process['error']}"
        external_conflict = bool(process.get("external_conflict"))
        if external_conflict:
            info += "\n\nAÇÃO: encerre o processo externo ou escolha outra porta."
        self.process_info_label.configure(text=info, text_color=RED if process.get("error") else MUTED)
        self.agent_info_label.configure(
            text=(
                f"Perfil: {agent.get('agent_env') or 'será gerado ao iniciar'}\n"
                f"Endpoint: {agent.get('endpoint') or '—'}\n"
                f"Compacção: {compact_int(agent.get('auto_compact_token_limit'))}\n"
                f"OpenCode: {agent.get('opencode_config') or 'aguardando ativação'}"
            )
        )
        self.start_button.configure(
            state="disabled" if running or external_conflict else "normal"
        )
        self.stop_button.configure(state="normal" if running else "disabled")
        self.mcp_switch.configure(
            state="disabled" if running or not BUNDLED_MCP_AVAILABLE else "normal"
        )

    # ---------------------------------------------------------- system events
    def _refresh_hardware(self) -> None:
        self._run_task(self.backend.refresh_hardware, self._hardware_refreshed, "Atualizando CPU, RAM e GPU…")

    def _apply_dynamic_swap(self) -> None:
        hardware = self.backend.hardware_snapshot()
        current = int(
            (int(hardware.get("swap_nvme_total_mb", 0)) + 1023) // 1024
        )
        size = max(int(hardware.get("swap_recommended_gib", 0)), current)
        if size <= 0:
            messagebox.showinfo(
                "Crono Matrix",
                "O modelo atual não precisa de swap NVMe adicional.",
                parent=self,
            )
            return
        self._run_task(
            lambda: self.backend.configure_nvme_swap("create", size),
            self._hardware_refreshed,
            f"Aguardando autenticação para aplicar {size} GiB de swap NVMe…",
        )

    def _toggle_auto_swap(self) -> None:
        enabled = self.backend.hardware_snapshot().get("auto_nvme_swap") != "y"
        self._run_task(
            lambda: self.backend.set_auto_nvme_swap(enabled),
            self._hardware_refreshed,
            "Atualizando política de swap…",
        )

    def _remove_nvme_swap(self) -> None:
        if not messagebox.askyesno(
            "Crono Matrix",
            "Remover o swap NVMe exclusivo do Crono Matrix?",
            parent=self,
        ):
            return
        self._run_task(
            lambda: self.backend.configure_nvme_swap("remove"),
            self._hardware_refreshed,
            "Aguardando autenticação para remover o swap NVMe…",
        )

    def _verify_model_updates(self) -> None:
        self._run_task(
            lambda: self.backend.start_model_update_check(),
            self._model_updates_done,
            "Verificando atualizações de modelos...",
        )

    def _model_updates_done(self, _result: Any) -> None:
        self._render_models(self.backend.models_snapshot())
        self._set_status("Verificação de atualizações concluída.")

    def _hardware_refreshed(self, hardware: dict) -> None:
        self._render_hardware(hardware)
        self._set_status("Telemetria atualizada; alterações manuais foram preservadas.")

    def _render_hardware(self, hw: dict) -> None:
        guard = hw.get("memory_guard", {})
        self.cpu_card.configure(text=(
            f"{hw['cpu_model']}\n\n"
            f"{hw['cpu_cores']} núcleos / {hw['cpu_threads']} threads · {hw['cpu_temp']} °C\n"
            f"RAM livre {hw['ram_avail_gb']:.1f} / {hw['ram_total_gb']:.1f} GiB\n"
            f"Guard C99: piso {int(hw.get('ram_reserve_mb', 2048)) / 1024:.1f} GiB · "
            f"{guard.get('last_action', 'aguardando modelo')}"
        ))
        self.gpu_card.configure(text=(
            f"{hw['gpu_model']}\n\n"
            f"VRAM livre {hw['gpu_vram_free_gb']} / {hw['gpu_vram_gb']} GiB · {hw['gpu_temp']} °C\n"
            f"Driver {hw['gpu_driver'] or '—'} · CUDA {hw['gpu_cuda'] or '—'}"
        ))
        cfg = self.backend.configuration_snapshot()
        self.storage_card.configure(text=(
            f"Storage {hw['storage_type']} · {hw['disk_free_gb']} GiB livres\n\n"
            f"ZRAM {hw['swap_zram_total_mb'] / 1024:.1f} GiB "
            f"(prio {hw.get('swap_zram_priority', '—')}) · "
            f"NVMe {hw['swap_nvme_total_mb'] / 1024:.1f} GiB "
            f"(prio {hw.get('swap_nvme_priority', '—')}) · "
            f"recomendado {hw['swap_recommended_gib']} GiB\n"
            f"{'PRIORIDADE INSEGURA: reaplique o swap NVMe.\n' if hw.get('swap_nvme_active') and not hw.get('swap_nvme_preferred') else ''}"
            f"{hw['swap_plan_reason']}\n\n"
            f"llama-server:\n{cfg['llama_server']}\n\n"
            f"llama-fit-params:\n{cfg['llama_fit_params']}"
        ))
        automatic = hw.get("auto_nvme_swap") == "y"
        self.swap_auto_button.configure(
            text=f"SWAP AUTOMÁTICO: {'ON' if automatic else 'OFF'}"
        )
        current_swap_gib = int(
            (int(hw.get("swap_nvme_total_mb", 0)) + 1023) // 1024
        )
        needs_growth = int(hw.get("swap_recommended_gib", 0)) > current_swap_gib
        needs_priority = bool(
            int(hw.get("swap_recommended_gib", 0)) > 0
            and hw.get("swap_nvme_active")
            and not hw.get("swap_nvme_preferred")
        )
        self.swap_apply_button.configure(
            state="normal" if needs_growth or needs_priority else "disabled",
            text=(
                f"APLICAR {hw['swap_recommended_gib']} GiB DINÂMICOS"
                if needs_growth else (
                    "CORRIGIR PRIORIDADE DO SWAP"
                    if needs_priority else "SWAP DINÂMICO ATENDIDO"
                )
            ),
        )
        self.swap_remove_button.configure(
            state="normal" if hw.get("swap_nvme_active") else "disabled"
        )
        self._render_memory_guard(hw.get("memory_guard"), force=True)

    def _render_memory_guard(self, guard: dict | None, force: bool = False) -> None:
        """Atualiza widgets; a coleta permanece exclusiva do processo C99."""
        mg = memory_guard_view(guard)
        render_key = tuple(mg.values())
        if not force and render_key == self._last_memory_guard_view:
            return
        self._last_memory_guard_view = render_key
        color = RED if mg["error"] else (AMBER if mg["pressure"] else TEXT)
        self.memory_guard_card.configure(text=(
            "Monitor nativo C99\n\n"
            f"Status: {mg['last_action']}\n"
            f"RAM disponível: {compact_int(mg['available_mb'])} MiB\n"
            f"Uso do cgroup: {compact_int(mg['current_mb'])} MiB\n"
            f"Aviso abaixo de: {compact_int(mg['trigger_mb'])} MiB\n"
            f"Pressão: {compact_int(mg['pressure_count'])} evento(s)\n"
            f"Erro: {mg['error'] or 'nenhum'}"
        ), text_color=color)
        self.memory_guard_card2.configure(text=(
            "Cgroup / Scope\n\n"
            f"Unidade: {mg['scope_unit']}\n"
            f"Fase: {mg['scope_phase']}\n"
            f"MemoryHigh: {mg['memory_high']}\n"
            f"Headroom: {compact_int(mg['scope_headroom_mb'])} MiB\n"
            "Política: observação; paginação pelo kernel/NVMe"
        ))

    # ---------------------------------------------------------------- helpers
    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.configure(text=text, text_color=RED if error else MUTED)

    @staticmethod
    def _replace_text(widget: ctk.CTkTextbox, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _on_close(self) -> None:
        if self.backend.is_running():
            leave = messagebox.askyesno(
                "Crono Matrix",
                "O llama-server está em execução. Fechar a interface e manter o servidor ativo?",
                parent=self,
            )
            if not leave:
                return
        self._closing = True
        self.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crono Matrix desktop local")
    parser.add_argument("--models-dir", default="", help="Diretório inicial dos GGUF")
    parser.add_argument("--llama-cpp-dir", default="", help="Diretório do llama.cpp local")
    parser.add_argument(
        "--legacy-tk", action="store_true",
        help="Abre a interface Tk antiga para compatibilidade temporária",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.legacy_tk:
        from launch_model_ui import main as legacy_main
        legacy_main()
        return
    state = LauncherWebState(models_dir=args.models_dir, llama_cpp_dir=args.llama_cpp_dir)
    eval_runner = EvalRunner()
    app = CronoDesktop(state, eval_runner=eval_runner)
    app.mainloop()


if __name__ == "__main__":
    main()

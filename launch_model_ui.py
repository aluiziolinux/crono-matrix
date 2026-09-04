#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface Tkinter do Crono llama-server launcher."""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import subprocess
import threading
import queue
import signal
import socket
import math
import ipaddress
import time as _time
import os
import sys
import re
import glob
import webbrowser
import json
import hashlib
import urllib.parse
import urllib.request
import shlex
import shutil
from pathlib import Path

from launch_model_core import (
    HF_API,
    HF_TRUSTED,
    HardwareInfo,
    HuggingFaceHub,
    LLAMA_FIT_PARAMS,
    LLAMA_SERVER,
    MCP_DIR,
    MCP_ENTRY,
    MCP_PORT,
    MEDIA_PATH,
    MODELS_DIR,
    ModelMetadata,
    OptimalParams,
    _find_companion,
    _gguf_total_size,
    _hf_base_model_name,
    _hf_display_name,
    _hf_fetch_json,
    _hf_format_bytes,
    _hf_format_params,
    _hf_headers,
    _hf_translate_lock,
    _hf_translate_ok,
    _hf_translator,
    _is_auxiliary_gguf,
    _requires_symmetric_kv,
    _is_secondary_shard,
    _server_supports_flag,
)


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MCP_DIR = PROJECT_ROOT / "mcp-crono-matrix"
LOCAL_MCP_ENTRY = LOCAL_MCP_DIR / "native_server.mjs"
LOCAL_MCP_WORKSPACE = LOCAL_MCP_DIR / "workspace"
LOCAL_MCP_MEMORY = LOCAL_MCP_DIR / "memory"
LOCAL_SNN_ENABLED = LOCAL_MCP_MEMORY / "snn" / "enabled.json"


# Paleta "Crono Matrix": a GUI desktop segue a identidade da interface web,
# mas continua sendo Tk puro e não carrega nenhum recurso remoto.
BG      = "#06110b"
BG2     = "#0a1b12"
BG3     = "#10271a"
BG4     = "#020905"
FG      = "#d5eadc"
FG2     = "#78a589"
BLUE    = "#5fffa0"       # nome mantido para reduzir churn no código legado
GREEN   = "#42f58d"
YELLOW  = "#e6c65c"
RED     = "#ff626d"
MAGENTA = "#80d8ff"
CYAN    = "#78ffc0"
TEAL    = "#55ffa0"
BORDER  = "#1a5132"
SEL     = "#145f36"

FM = ("DejaVu Sans Mono", 10)
FS = ("DejaVu Sans Mono", 9)
FB = ("DejaVu Sans Mono", 11, "bold")
FT = ("DejaVu Sans Mono", 13, "bold")
FH = ("DejaVu Sans Mono", 16, "bold")


# ════════════════════════════════════════════════════════════
#   APLICAÇÃO PRINCIPAL
# ════════════════════════════════════════════════════════════
class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.settings_file = Path.home() / ".config" / "crono-launcher" / "settings.json"
        settings = self._load_settings()
        self.root        = root
        self.hw          = HardwareInfo()
        self.meta        = None
        self.opt         = None
        self.proc        = None
        self.mcp_proc    = None
        self.mcp_enabled = tk.BooleanVar(value=bool(settings.get("mcp_enabled", False)))
        self.mcp_port    = tk.StringVar(value=str(settings.get("mcp_port", MCP_PORT)))
        self.mcp_auto    = tk.BooleanVar(value=bool(settings.get("mcp_auto", True)))
        launch_mode = settings.get("launch_mode", "launch")
        self.launch_mode = tk.StringVar(
            value=launch_mode if launch_mode in {"launch", "connect"} else "launch"
        )
        self.conn_host   = tk.StringVar(value=settings.get("conn_host", "127.0.0.1"))
        self.conn_port   = tk.StringVar(value=str(settings.get("conn_port", "8080")))
        self.connected   = False
        self.model_path  = ""
        self._models     = []
        self._pvars      = {}   # param StringVars
        self.models_dir_var = tk.StringVar(value=settings.get("models_dir", MODELS_DIR))
        self._last_model_path = settings.get("last_model_path", "")
        self.hf = HuggingFaceHub()
        self.hf_search_results = []
        self.hf_ggufs = []
        self.hf_selected = None  # {user, repo, file}
        self.hf_readme_raw = None  # raw README text for re-translation
        self.hf_readme_is_translated = False
        self._ui_queue = queue.Queue()
        self._closing = False
        self._hw_request = 0
        self._hardware_ready = False
        self._meta_request = 0
        self._hf_request = 0
        self._hf_detail_request = 0
        self._hf_render_request = 0
        self._models_request = 0
        self._connect_request = 0
        self._test_request = 0
        self._resolve_request = 0
        self._recalc_request = 0
        self._recalc_after = None
        self._updating_params = False
        self._recalc_running = False
        self._runtime_request = 0
        self._download_active = False
        self._download_cancel = threading.Event()
        self._download_thread = None

        self._setup_window()
        self._setup_styles()
        self._build_ui()
        self._on_mode_change()
        self.root.after(50, self._drain_ui_queue)
        # Inicia detecção de hardware em background
        self._start_hw_detection()

    def _load_settings(self) -> dict:
        try:
            with self.settings_file.open(encoding="utf-8") as handle:
                settings = json.load(handle)
            return settings if isinstance(settings, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _load_snn_enabled() -> bool:
        try:
            value = json.loads(LOCAL_SNN_ENABLED.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("enabled"), bool):
                return value["enabled"]
        except (OSError, ValueError, TypeError):
            pass
        return False

    def _save_settings(self) -> None:
        settings = self._load_settings()
        settings.update({
            "models_dir": self.models_dir_var.get(),
            "last_model_path": self.model_path,
            "mcp_enabled": self.mcp_enabled.get(),
            "mcp_auto": self.mcp_auto.get(),
            "mcp_port": self.mcp_port.get(),
            "launch_mode": self.launch_mode.get(),
            "conn_host": self.conn_host.get(),
            "conn_port": self.conn_port.get(),
        })
        temp_path = self.settings_file.with_name(self.settings_file.name + ".tmp")
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(settings, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_path, self.settings_file)
        except OSError:
            try:
                temp_path.unlink()
            except OSError:
                pass

    def _post_ui(self, callback, *args) -> None:
        if not self._closing:
            self._ui_queue.put((callback, args))

    def _drain_ui_queue(self) -> None:
        if self._closing:
            return
        while True:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except (tk.TclError, RuntimeError):
                if not self._closing:
                    raise
        self.root.after(50, self._drain_ui_queue)

    # ── Janela ───────────────────────────────────────────────
    def _setup_window(self) -> None:
        self.root.title("Crono Matrix — llama.cpp local control plane")
        self.root.geometry("1320x860")
        self.root.minsize(1080, 720)
        self.root.configure(bg=BG)
        try:
            self.root.tk.call("wm", "iconphoto", self.root._w,
                              tk.PhotoImage(data="R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="))
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            if messagebox.askyesno("Servidor ativo",
                                   "O servidor está em execução.\nEncerrar mesmo assim?"):
                pass
            else:
                return
        self._save_settings()
        self._closing = True
        self._download_cancel.set()
        self.hf.cancel_downloads()
        if self._download_thread and self._download_thread.is_alive():
            self._download_thread.join(timeout=35)
        self._terminate_process(self.proc)
        self._terminate_process(self.mcp_proc)
        self.root.destroy()

    # ── Estilos ttk ──────────────────────────────────────────
    def _setup_styles(self) -> None:
        s = ttk.Style()
        s.theme_use("clam")
        base = dict(background=BG, foreground=FG, font=FM,
                    fieldbackground=BG3, insertcolor=FG, selectbackground=SEL)
        s.configure(".", **base)
        # Notebook
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                    padding=[18, 10], font=FM, borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", BLUE)])
        # Frames
        s.configure("TFrame",      background=BG)
        s.configure("Card.TFrame", background=BG2)
        s.configure("Dark.TFrame", background=BG4)
        # Labels
        s.configure("TLabel",        background=BG,  foreground=FG,  font=FM)
        s.configure("S.TLabel",      background=BG,  foreground=FG2, font=FS)
        s.configure("Card.TLabel",   background=BG2, foreground=FG,  font=FM)
        s.configure("CardS.TLabel",  background=BG2, foreground=FG2, font=FS)
        s.configure("Cyan.TLabel",   background=BG2, foreground=CYAN, font=FB)
        s.configure("Blue.TLabel",   background=BG,  foreground=BLUE, font=FM)
        s.configure("Green.TLabel",  background=BG,  foreground=GREEN,font=FM)
        s.configure("Yellow.TLabel", background=BG,  foreground=YELLOW,font=FM)
        s.configure("Red.TLabel",    background=BG,  foreground=RED,  font=FM)
        s.configure("Mag.TLabel",    background=BG,  foreground=MAGENTA,font=FM)
        s.configure("Dim.TLabel",    background=BG,  foreground=FG2,  font=FS)
        # Entries / Combobox
        s.configure("TEntry",    fieldbackground=BG3, foreground=FG, font=FM,
                    insertcolor=FG, borderwidth=0, relief="flat")
        s.configure("TCombobox", fieldbackground=BG3, foreground=FG, font=FM,
                    selectbackground=SEL, borderwidth=0)
        s.map("TCombobox", fieldbackground=[("readonly", BG3)],
              foreground=[("readonly", FG)])
        # Scrollbar
        s.configure("TScrollbar", background=BG3, troughcolor=BG,
                    arrowcolor=FG2, borderwidth=0, relief="flat")
        # Checkbutton
        s.configure("TCheckbutton", background=BG, foreground=FG, font=FM)
        s.map("TCheckbutton", background=[("active", BG)])
        # Buttons
        s.configure("TButton", background=BG3, foreground=FG, font=FM,
                    padding=[12, 6], relief="flat", borderwidth=0)
        s.map("TButton", background=[("active", "#2d333b")])
        s.configure("Green.TButton", background="#15733e", foreground="#e5fff0",
                    font=FB, padding=[16, 8])
        s.map("Green.TButton", background=[("active", "#1ca759")])
        s.configure("Red.TButton",   background="#5d1f29", foreground="#ffd7da",
                    font=FB, padding=[14, 7])
        s.map("Red.TButton",   background=[("active", "#9b2e3a")])
        s.configure("Blue.TButton",  background="#123d28", foreground=BLUE,
                    font=FM, padding=[12, 6])
        s.map("Blue.TButton",  background=[("active", "#1c5d3d")])
        s.configure("Gray.TButton",  background=BG3, foreground=FG2,
                    font=FM, padding=[10, 5])
        s.map("Gray.TButton",  background=[("active", "#2d333b")])
        # Separator
        s.configure("TSeparator", background=BORDER)
        # Progressbar
        s.configure("TProgressbar", background=BLUE, troughcolor=BG3,
                    borderwidth=0, thickness=3)

    # ── Layout principal ─────────────────────────────────────
    @staticmethod
    def _local_build_label() -> str:
        """Describe the selected local binary without network access."""
        try:
            completed = subprocess.run(
                [LLAMA_SERVER, "--version"], capture_output=True, text=True,
                timeout=5, check=False,
            )
            output = completed.stdout + "\n" + completed.stderr
            match = re.search(r"version:\s*([^\n]+)", output)
            if match:
                return match.group(1).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return Path(LLAMA_SERVER).name

    def _build_ui(self) -> None:
        # Header
        hdr = tk.Frame(self.root, bg=BG4, height=82, highlightbackground=BORDER,
                       highlightthickness=1)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        brand = tk.Frame(hdr, bg=BG4)
        brand.pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(brand, text="C://  CRONO MATRIX", bg=BG4, fg=GREEN,
                 font=FH).pack(anchor="w")
        tk.Label(brand, text="LLAMA.CPP LOCAL CONTROL PLANE  //  OFFLINE-FIRST",
                 bg=BG4, fg=FG2, font=FS).pack(anchor="w", pady=(2, 0))

        runtime = tk.Frame(hdr, bg=BG4)
        runtime.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=9)
        self.header_state_var = tk.StringVar(value="◇ IDLE")
        self.header_model_var = tk.StringVar(value="NENHUM MODELO")
        tk.Label(runtime, textvariable=self.header_state_var, bg=BG4,
                 fg=GREEN, font=FB, anchor="e").pack(anchor="e")
        tk.Label(runtime, textvariable=self.header_model_var, bg=BG4,
                 fg=FG, font=FS, anchor="e").pack(anchor="e", pady=(4, 0))
        tk.Label(runtime, text=self._local_build_label(), bg=BG4,
                 fg=FG2, font=FS, anchor="e").pack(anchor="e")

        # Barra de progresso (usada durante detecção)
        self.prog = ttk.Progressbar(self.root, mode="indeterminate",
                                    style="TProgressbar", length=200)
        self.prog.pack(fill=tk.X, side=tk.TOP, padx=0, pady=0)

        # Status bar
        self.status_var = tk.StringVar(value="Detectando hardware...")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=BG4, fg=FG2, font=FS, anchor="w", pady=4, padx=12
                 ).pack(fill=tk.X, side=tk.BOTTOM)

        # Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_hw       = ttk.Frame(self.nb)
        self.tab_model    = ttk.Frame(self.nb)
        self.tab_hf       = ttk.Frame(self.nb)
        self.tab_params   = ttk.Frame(self.nb)
        self.tab_launch   = ttk.Frame(self.nb)

        self.nb.add(self.tab_hw,       text=" [01] SISTEMA ")
        self.nb.add(self.tab_model,    text=" [02] MODELOS LOCAIS ")
        self.nb.add(self.tab_hf,       text=" [03] HUGGING FACE (ONLINE) ")
        self.nb.add(self.tab_params,   text=" [04] INFERÊNCIA ")
        self.nb.add(self.tab_launch,   text=" [05] PROCESSO ")

        self._build_tab_hw()
        self._build_tab_model()
        self._build_tab_hf()
        self._build_tab_params()
        self._build_tab_launch()

    # ════════════════════════════════════════════════════════
    #   TAB: SISTEMA
    # ════════════════════════════════════════════════════════
    def _build_tab_hw(self) -> None:
        f = self.tab_hw
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        # Cartão CPU
        cc = self._card(f, "⚡  CPU")
        cc.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self.v_cpu_model   = self._kv(cc, "Modelo",           width=18)
        self.v_cpu_cores   = self._kv(cc, "Cores / Threads",  width=18)
        self.v_cpu_temp    = self._kv(cc, "Temperatura",      width=18)

        # Cartão RAM
        rc = self._card(f, "🧠  Memória RAM")
        rc.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")
        self.v_ram_total   = self._kv(rc, "Total",            width=18)
        self.v_ram_avail   = self._kv(rc, "Disponível",       width=18)

        # Cartão GPU
        gc = self._card(f, "🎮  GPU")
        gc.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")
        self.v_gpu_model   = self._kv(gc, "Modelo",           width=18)
        self.v_gpu_vram    = self._kv(gc, "VRAM Total",       width=18)
        self.v_gpu_free    = self._kv(gc, "VRAM Livre",       width=18)
        self.v_gpu_temp    = self._kv(gc, "Temperatura",      width=18)
        self.v_gpu_driver  = self._kv(gc, "Driver",           width=18)
        self.v_gpu_cuda    = self._kv(gc, "Compute Cap",      width=18)

        # Cartão Storage
        sc = self._card(f, "💾  Storage")
        sc.grid(row=1, column=1, padx=12, pady=12, sticky="nsew")
        self.v_storage     = self._kv(sc, "Tipo",             width=18)
        self.v_disk_free   = self._kv(sc, "Espaço livre",     width=18)

        # Botão atualizar
        bf = ttk.Frame(f)
        bf.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(bf, text="↻   Atualizar Hardware", style="Blue.TButton",
                   command=self._start_hw_detection
                   ).pack()

    def _card(self, parent, title: str) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=14)
        tk.Label(frame, text=title, bg=BG2, fg=CYAN, font=FB).pack(anchor="w", pady=(0, 8))
        tk.Frame(frame, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 8))
        return frame

    def _kv(self, parent, label: str, width: int = 22) -> tk.StringVar:
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=f"{label}:", bg=BG2, fg=FG2, font=FS,
                 width=width, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value="Detectando...")
        tk.Label(row, textvariable=var, bg=BG2, fg=FG, font=FM).pack(side=tk.LEFT)
        return var

    def _start_hw_detection(self) -> None:
        self._hw_request += 1
        self._hardware_ready = False
        request_id = self._hw_request
        self.prog.start(8)
        threading.Thread(target=self._hw_thread, args=(request_id,), daemon=True).start()

    def _hw_thread(self, request_id: int) -> None:
        hw = HardwareInfo()
        hw.detect()
        self._post_ui(self._hw_done, request_id, hw)

    def _hw_done(self, request_id: int, hw: HardwareInfo):
        if request_id != self._hw_request:
            return
        self.prog.stop()
        self.hw = hw
        self._hardware_ready = True
        self.v_cpu_model.set(hw.cpu_model[:52])
        self.v_cpu_cores.set(f"{hw.cpu_cores} físicos / {hw.cpu_threads} lógicos")
        self.v_cpu_temp.set(f"{hw.cpu_temp}°C" if hw.cpu_temp else "N/D")
        self.v_ram_total.set(f"{hw.ram_total_gb:.1f} GB")
        self.v_ram_avail.set(f"{hw.ram_avail_gb:.1f} GB disponível")
        if hw.gpu_detected:
            self.v_gpu_model.set(hw.gpu_model)
            self.v_gpu_vram.set(f"{hw.gpu_vram_gb} GB")
            self.v_gpu_free.set(f"{hw.gpu_vram_free_gb} GB livres")
            self.v_gpu_temp.set(f"{hw.gpu_temp}°C")
            self.v_gpu_driver.set(hw.gpu_driver or "N/D")
            self.v_gpu_cuda.set(hw.gpu_cuda or "N/D")
        else:
            for v in (self.v_gpu_model, self.v_gpu_vram, self.v_gpu_free,
                      self.v_gpu_temp, self.v_gpu_driver, self.v_gpu_cuda):
                v.set("—  (GPU não detectada)")
        self.v_storage.set(hw.storage_type)
        self.v_disk_free.set(f"{hw.disk_free_gb} GB")
        if self.meta and not (self.proc and self.proc.poll() is None):
            self.status_var.set("✔  Hardware atualizado; recalculando o plano sem apagar suas edições...")
            self._schedule_memory_recalc("hardware", delay=0)
        else:
            self.status_var.set("✔  Hardware detectado — selecione um modelo na aba Modelo.")

    # ════════════════════════════════════════════════════════
    #   TAB: MODELO
    # ════════════════════════════════════════════════════════
    def _build_tab_model(self) -> None:
        f = self.tab_model
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)

        # Cabeçalho com seletor de diretório
        top = tk.Frame(f, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=10)
        tk.Label(top, text="📁  Selecione o modelo GGUF", bg=BG, fg=BLUE, font=FB).pack(side=tk.LEFT)

        dir_frame = tk.Frame(top, bg=BG)
        dir_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=14)
        tk.Label(dir_frame, text="Diretório:", bg=BG, fg=FG2, font=FS).pack(side=tk.LEFT)
        dir_entry = ttk.Entry(dir_frame, textvariable=self.models_dir_var, font=FS)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(dir_frame, text="📂  Browse", style="Gray.TButton",
                   command=self._browse_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_frame, text="↻", style="Gray.TButton",
                   command=self._load_models, width=3).pack(side=tk.LEFT)

        # Listbox + scrollbar
        lf = tk.Frame(f, bg=BG2, bd=1, relief="flat")
        lf.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self.model_lb = tk.Listbox(
            lf, bg=BG2, fg=FG, selectbackground=SEL, selectforeground=FG,
            font=FM, relief="flat", borderwidth=0, activestyle="none",
            highlightthickness=0
        )
        self.model_lb.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.model_lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.model_lb.configure(yscrollcommand=sb.set)
        self.model_lb.bind("<<ListboxSelect>>", self._on_model_hover)
        self.model_lb.bind("<Double-Button-1>", lambda e: self._model_confirm())

        # Info rápida do modelo selecionado
        self.model_info_var = tk.StringVar(value="Selecione um modelo acima.")
        info_frame = tk.Frame(f, bg=BG2, padx=14, pady=8)
        info_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=4)
        tk.Label(info_frame, textvariable=self.model_info_var,
                 bg=BG2, fg=FG2, font=FS, anchor="w", justify="left",
                 wraplength=900).pack(fill=tk.X)

        # Botão confirmar
        bf = ttk.Frame(f)
        bf.grid(row=3, column=0, pady=10)
        ttk.Button(bf,
                   text="✔   Selecionar Modelo  →  Calcular Parâmetros",
                   style="Green.TButton",
                   command=self._model_confirm).pack()

        self._load_models()

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(initialdir=self.models_dir_var.get(),
                                     title="Selecione a pasta com modelos GGUF")
        if d:
            self.models_dir_var.set(d)
            self._load_models()

    def _load_models(self) -> None:
        self.model_lb.delete(0, tk.END)
        d = self.models_dir_var.get()
        if not os.path.isdir(d):
            self.model_lb.insert(tk.END, f"  ⚠  Diretório não encontrado: {d}")
            self.status_var.set("Diretório inválido.")
            return
        self._models_request += 1
        request_id = self._models_request
        self.model_lb.insert(tk.END, "  Procurando modelos...")
        self.status_var.set("Procurando modelos GGUF...")
        threading.Thread(
            target=self._load_models_thread, args=(request_id, d), daemon=True
        ).start()

    def _load_models_thread(self, request_id: int, directory: str) -> None:
        raw = (glob.glob(f"{directory}/*.gguf") +
               glob.glob(f"{directory}/**/*.gguf", recursive=True))
        candidates = sorted(set(
            m for m in raw if not _is_auxiliary_gguf(m) and not _is_secondary_shard(m)
        ))
        models = []
        rows = []
        for model in candidates:
            try:
                size = _gguf_total_size(model) / 1073741824
                models.append(model)
                rows.append(f"  {os.path.basename(model):<62}  {size:.2f} GB")
            except (OSError, ValueError):
                continue
        self._post_ui(self._load_models_done, request_id, directory, models, rows)

    def _load_models_done(self, request_id: int, directory: str, models: list, rows: list) -> None:
        if request_id != self._models_request or directory != self.models_dir_var.get():
            return
        self.model_lb.delete(0, tk.END)
        self._models = models
        for line in rows:
            self.model_lb.insert(tk.END, line)
        if self._last_model_path in self._models:
            index = self._models.index(self._last_model_path)
            self.model_lb.selection_set(index)
            self.model_lb.see(index)
            self._on_model_hover()
        if not self._models:
            self.model_lb.insert(tk.END, "  ⚠  Nenhum .gguf encontrado no diretório configurado.")
        self.status_var.set(f"{len(self._models)} modelo(s) encontrado(s).")

    def _on_model_hover(self, _event=None) -> None:
        sel = self.model_lb.curselection()
        if not sel or sel[0] >= len(self._models):
            return
        path  = self._models[sel[0]]
        fname = os.path.basename(path)
        try:
            sz = os.path.getsize(path) / 1073741824 if os.path.isfile(path) else 0.0
        except OSError:
            sz = 0.0
        mq    = re.search(r"(?i)(Q[0-9]_[A-Z0-9_]+|IQ[0-9][A-Z0-9_]*|BF16|F16|F32|FP16)", fname)
        mp    = re.search(r"(\d+\.?\d*[BbMm])", fname)
        d     = os.path.dirname(path)
        mm_path = _find_companion(path, "mmproj")
        vc_path = _find_companion(path, "vocoder")
        self.model_info_var.set(
            f"Arquivo: {fname}  |  Tamanho: {sz:.2f} GB  |  "
            f"Quant: {mq.group(1) if mq else '?'}  |  Params: {mp.group(1).upper() if mp else '?'}\n"
            f"mmproj: {'✔ ' + os.path.basename(mm_path) if mm_path else '✗ nenhum'}   "
            f"vocoder: {'✔ ' + os.path.basename(vc_path) if vc_path else '✗ nenhum'}"
        )

    def _model_confirm(self) -> None:
        sel = self.model_lb.curselection()
        if not sel or sel[0] >= len(self._models):
            messagebox.showwarning("Aviso", "Selecione um modelo antes de continuar.")
            return
        if not self._hardware_ready:
            messagebox.showinfo(
                "Hardware em detecção",
                "Aguarde a detecção de hardware terminar antes de calcular os parâmetros.",
            )
            return
        self.model_path = self._models[sel[0]]
        self._last_model_path = self.model_path
        self._meta_request += 1
        request_id = self._meta_request
        model_path = self.model_path
        self.status_var.set(f"Carregando metadados: {os.path.basename(self.model_path)}...")
        self.prog.start(8)
        threading.Thread(target=self._meta_thread, args=(request_id, model_path), daemon=True).start()

    def _meta_thread(self, request_id: int, model_path: str) -> None:
        meta = ModelMetadata()
        error = ""
        try:
            meta.load(model_path)
        except Exception as exc:
            error = str(exc)
        self._post_ui(self._meta_done, request_id, model_path, meta, error)

    def _meta_done(self, request_id: int, model_path: str, meta: ModelMetadata,
                   error: str) -> None:
        if request_id != self._meta_request or model_path != self.model_path:
            return
        self.meta = meta
        self.opt = OptimalParams(self.hw, meta)
        self.opt.calculate()
        self.prog.stop()
        mt = self.meta; opt = self.opt
        mtp_tag = "  |  MTP: ✔" if mt.has_mtp else ""
        self.status_var.set(
            f"✔  {os.path.basename(self.model_path)}  |  "
            f"{mt.size_gb_str} GB  |  ctx: {mt.ctx_max}  |  "
            f"layers: {mt.layers}  |  arch: {mt.arch}  |  quant: {mt.quant}"
            f"{mtp_tag}"
        )
        self.header_model_var.set(os.path.basename(self.model_path))
        self._fill_params()
        self.nb.select(self.tab_params)
        warnings = [value for value in (error, meta.metadata_error, meta.profile_error) if value]
        if warnings:
            messagebox.showwarning(
                "Metadados parciais",
                "Alguns metadados não puderam ser lidos:\n" + "\n".join(warnings),
            )

    # ════════════════════════════════════════════════════════
    #   TAB: HUGGINGFACE
    # ════════════════════════════════════════════════════════
    def _build_tab_hf(self) -> None:
        f = self.tab_hf
        f.rowconfigure(2, weight=1)
        f.columnconfigure(0, weight=2)
        f.columnconfigure(1, weight=1)

        # ── Topo: busca + atalhos ────────────────────────────
        top = tk.Frame(f, bg=BG)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 2))
        tk.Label(top, text="🌐  HuggingFace Hub",
                 bg=BG, fg=BLUE, font=FB).pack(side=tk.LEFT)

        self.hf_search_var = tk.StringVar()
        search_entry = ttk.Entry(top, textvariable=self.hf_search_var, width=36)
        search_entry.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        search_entry.bind("<Return>", lambda e: self._hf_do_search())
        ttk.Button(top, text="🔍  Buscar", style="Blue.TButton",
                   command=self._hf_do_search).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="🔥  Trending", style="Gray.TButton",
                   command=lambda: self._hf_load_trending("text-generation")).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="📷  Vision", style="Gray.TButton",
                   command=lambda: self._hf_load_trending("image-text-to-text")).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="🧠  Embed", style="Gray.TButton",
                   command=lambda: self._hf_load_trending("sentence-similarity")).pack(side=tk.LEFT, padx=2)

        # ── Painel esquerdo: resultados ──────────────────────
        left = tk.Frame(f, bg=BG2, bd=1, relief="flat")
        left.grid(row=2, column=0, sticky="nsew", padx=(12, 6), pady=4)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self.hf_result_header = tk.StringVar(value="🔥  Trending — text-generation")
        tk.Label(left, textvariable=self.hf_result_header,
                 bg=BG2, fg=CYAN, font=FS
                 ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))

        self.hf_result_lb = tk.Listbox(
            left, bg=BG2, fg=FG, selectbackground=SEL, selectforeground=FG,
            font=FM, relief="flat", borderwidth=0, activestyle="none",
            highlightthickness=0
        )
        self.hf_result_lb.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
        sb1 = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.hf_result_lb.yview)
        sb1.grid(row=1, column=1, sticky="ns")
        self.hf_result_lb.configure(yscrollcommand=sb1.set)
        self.hf_result_lb.bind("<<ListboxSelect>>", self._hf_on_result_select)

        # ── Painel direito: info + README + arquivos ─────────
        right = tk.Frame(f, bg=BG2, bd=1, relief="flat")
        right.grid(row=2, column=1, sticky="nsew", padx=(6, 12), pady=4)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        # Info compacta no topo
        self.hf_info_var = tk.StringVar(value="Selecione um modelo à esquerda")
        tk.Label(right, textvariable=self.hf_info_var,
                 bg=BG2, fg=FG, font=FM, wraplength=450, justify="left",
                 anchor="nw"
                 ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))

        # Notebook interno: README | Arquivos GGUF
        right_nb = ttk.Notebook(right)
        right_nb.grid(row=3, column=0, sticky="nsew", padx=4, pady=4)

        # ── Aba README ───────────────────────────────────────
        readme_frame = tk.Frame(right_nb, bg=BG4)
        readme_frame.rowconfigure(1, weight=1)
        readme_frame.columnconfigure(0, weight=1)

        readme_header = tk.Frame(readme_frame, bg=BG4)
        readme_header.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 0))

        self.hf_translate_var = tk.BooleanVar(value=True)
        self.hf_translate_btn = ttk.Checkbutton(
            readme_header, text="🌐  Traduzir para pt-BR",
            variable=self.hf_translate_var,
            command=self._hf_toggle_translate,
            style="Toolbutton"
        )
        self.hf_translate_btn.pack(side=tk.LEFT)

        if not _hf_translate_ok:
            self.hf_translate_btn.configure(state=tk.DISABLED)
            tk.Label(readme_header, text="(pip install deep-translator)",
                     bg=BG4, fg=YELLOW, font=FS).pack(side=tk.LEFT, padx=6)

        self.hf_readme_txt = scrolledtext.ScrolledText(
            readme_frame, bg=BG4, fg=FG, font=FS, relief="flat",
            wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0
        )
        self.hf_readme_txt.grid(row=1, column=0, sticky="nsew")
        self.hf_readme_txt.tag_configure("h1", foreground=BLUE, font=FB)
        self.hf_readme_txt.tag_configure("h2", foreground=CYAN, font=FB)
        self.hf_readme_txt.tag_configure("code", foreground=GREEN)
        self.hf_readme_txt.tag_configure("dim", foreground=FG2)
        self.hf_readme_txt.tag_configure("bold", font=FB)

        right_nb.add(readme_frame, text="  📖  README  ")

        # ── Aba Arquivos GGUF ────────────────────────────────
        files_frame = tk.Frame(right_nb, bg=BG2)
        files_frame.rowconfigure(1, weight=1)
        files_frame.columnconfigure(0, weight=1)

        self.hf_gguf_meta_var = tk.StringVar(
            value="Selecione um repositório GGUF para ver os metadados."
        )
        tk.Label(
            files_frame, textvariable=self.hf_gguf_meta_var,
            bg=BG3, fg=CYAN, font=FS, anchor="w", justify="left",
            padx=10, pady=7,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 0))

        self.hf_files_lb = tk.Listbox(
            files_frame, bg=BG2, fg=FG, selectbackground=SEL, selectforeground=FG,
            font=FM, relief="flat", borderwidth=0, activestyle="none",
            highlightthickness=0
        )
        self.hf_files_lb.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        sb2 = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.hf_files_lb.yview)
        sb2.grid(row=1, column=1, sticky="ns")
        self.hf_files_lb.configure(yscrollcommand=sb2.set)
        self.hf_files_lb.bind("<<ListboxSelect>>", self._hf_on_file_select)

        right_nb.add(files_frame, text="  📄  GGUF Files  ")

        # ── Barra inferior: progresso + botões ───────────────
        bot = tk.Frame(f, bg=BG)
        bot.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        bot.columnconfigure(0, weight=1)

        self.hf_prog_var = tk.StringVar(value="")
        tk.Label(bot, textvariable=self.hf_prog_var,
                 bg=BG, fg=FG2, font=FS).grid(row=0, column=0, sticky="w")

        btn_frame = tk.Frame(bot, bg=BG)
        btn_frame.grid(row=0, column=1, sticky="e")

        ttk.Button(btn_frame, text="📥  Baixar Selecionado",
                   style="Green.TButton",
                   command=self._hf_download).pack(side=tk.LEFT, padx=4)

        self.hf_cancel_btn = ttk.Button(
            btn_frame, text="Cancelar Download", style="Red.TButton",
            command=self._hf_cancel_download, state=tk.DISABLED,
        )
        self.hf_cancel_btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_frame, text="🔎  Resolver .gguf",
                   style="Blue.TButton",
                   command=self._hf_resolve_file).pack(side=tk.LEFT, padx=4)

        # ── Status ──────────────────────────────────────────
        self.hf_status_var = tk.StringVar(value="Carregando modelos populares...")
        tk.Label(f, textvariable=self.hf_status_var,
                 bg=BG4, fg=FG2, font=FS, anchor="w", padx=16, pady=2
                 ).grid(row=5, column=0, columnspan=2, sticky="ew")

        # ── Carrega trending ao abrir a aba ───────────────────
        self.nb.bind("<<NotebookTabChanged>>", self._hf_on_tab_select)
        self.root.after(500, lambda: self._hf_load_trending("text-generation"))

    # ── HF Tab select: carrega trending na primeira vez ──────
    def _hf_on_tab_select(self, _e=None) -> None:
        if self.nb.index(self.nb.select()) != 2:  # índice da aba HF
            return
        if not self.hf_search_results:
            self.root.after(100, lambda: self._hf_load_trending("text-generation"))

    # ── Trending / Top models ─────────────────────────────
    def _hf_load_trending(self, pipeline: str = "text-generation") -> None:
        labels = {
            "text-generation": "🔥  Trending — LLM",
            "image-text-to-text": "📷  Trending — Vision",
            "sentence-similarity": "🧠  Trending — Embeddings",
        }
        self.hf_result_header.set(labels.get(pipeline, f"📊  Trending — {pipeline}"))
        self.hf_result_lb.delete(0, tk.END)
        self.hf_result_lb.insert(tk.END, "  Carregando...")
        self.hf_status_var.set(f"Carregando {labels.get(pipeline, pipeline)}...")
        self._hf_invalidate_details()
        self._hf_request += 1
        request_id = self._hf_request
        threading.Thread(
            target=self._hf_trending_thread, args=(request_id, pipeline), daemon=True
        ).start()

    def _hf_trending_thread(self, request_id: int, pipeline: str) -> None:
        try:
            data = _hf_fetch_json(
                f"{HF_API}/models?pipeline_tag={pipeline}"
                f"&sort=downloads&direction=-1&full=true&limit=50"
            )
            self._post_ui(self._hf_search_done, request_id, data)
        except Exception as e:
            self._post_ui(self._hf_search_error, request_id, f"❌  Erro: {e}")

    # ── HF Search ──────────────────────────────────────────
    def _hf_do_search(self) -> None:
        term = self.hf_search_var.get().strip()
        if not term:
            return
        self.hf_result_header.set(f"🔍  Resultados para \"{term}\"")
        self.hf_result_lb.delete(0, tk.END)
        self.hf_result_lb.insert(tk.END, "  Buscando...")
        self.hf_files_lb.delete(0, tk.END)
        self.hf_ggufs = []
        self._hf_clear_readme()
        self.hf_info_var.set("Buscando...")
        self.hf_status_var.set(f"🔍  Buscando \"{term}\"...")
        self._hf_invalidate_details()
        self._hf_request += 1
        request_id = self._hf_request
        threading.Thread(target=self._hf_search_thread, args=(request_id, term), daemon=True).start()

    def _hf_invalidate_details(self) -> None:
        self._hf_detail_request += 1
        self._hf_render_request += 1
        self.hf_selected = None
        self.hf_search_results = []
        self.hf_ggufs = []
        self.hf_readme_raw = None
        self.hf_files_lb.delete(0, tk.END)
        self._hf_clear_readme()
        self.hf_info_var.set("Selecione um modelo à esquerda")
        self.hf_gguf_meta_var.set(
            "Selecione um repositório GGUF para ver os metadados."
        )

    def _hf_search_thread(self, request_id: int, term: str) -> None:
        try:
            direct = self.hf.search_by_url_or_id(term)
            if direct:
                info = self.hf.model_info(
                    direct["user"], direct["repo"], direct.get("revision", "main")
                )
                info["_crono_file"] = direct.get("file")
                info["_crono_revision"] = direct.get("revision", "main")
                results = [info]
            else:
                results = self.hf.search(term)
            self._post_ui(self._hf_search_done, request_id, results)
        except Exception as e:
            self._post_ui(self._hf_search_error, request_id, f"❌  Erro na busca: {e}")

    def _hf_search_error(self, request_id: int, message: str) -> None:
        if request_id == self._hf_request:
            self.hf_status_var.set(message)

    def _hf_search_done(self, request_id: int, results: list) -> None:
        if request_id != self._hf_request:
            return
        self.hf_search_results = results
        self.hf_result_lb.delete(0, tk.END)
        if not results:
            self.hf_result_lb.insert(tk.END, "  (nenhum resultado)")
            self.hf_status_var.set("Nenhum resultado encontrado.")
            return
        for r in results:
            downloads = r.get("downloads", 0)
            likes = r.get("likes", 0)
            d_str = f"{downloads//1000}k" if downloads >= 1000 else str(downloads)
            l_str = f"{likes//1000}k" if likes >= 1000 else str(likes)
            tag = r.get("pipeline_tag", "")
            label = f"  {r['id']:<52}  ⬇{d_str:>6}  ❤{l_str:>4}"
            self.hf_result_lb.insert(tk.END, label)
        self.hf_status_var.set(f"✔  {len(results)} resultado(s)   (⬇ downloads  ❤ likes)")

    def _hf_on_result_select(self, _e=None) -> None:
        sel = self.hf_result_lb.curselection()
        if not sel or sel[0] >= len(self.hf_search_results):
            return
        result = self.hf_search_results[sel[0]]
        self.hf_files_lb.delete(0, tk.END)
        self.hf_ggufs = []
        self.hf_files_lb.insert(tk.END, "  Carregando...")
        self.hf_gguf_meta_var.set("Carregando metadados GGUF...")
        self._hf_clear_readme()
        self.hf_readme_txt.configure(state=tk.NORMAL)
        self.hf_readme_txt.insert(tk.END, "  Carregando README...\n", "dim")
        self.hf_readme_txt.configure(state=tk.DISABLED)

        user, repo = result["id"].split("/", 1)
        self.hf_selected = {"user": user, "repo": repo, "file": None,
                            "downloads": result.get("downloads", 0),
                            "likes": result.get("likes", 0),
                            "revision": result.get("_crono_revision", "main")}
        self._hf_detail_request += 1
        request_id = self._hf_detail_request
        translate = self.hf_translate_var.get() and _hf_translate_ok
        threading.Thread(
            target=self._hf_load_files_thread,
            args=(request_id, user, repo, translate, result.get("_crono_file"),
                  result.get("_crono_revision", "main")), daemon=True
        ).start()

    def _hf_clear_readme(self) -> None:
        self.hf_readme_txt.configure(state=tk.NORMAL)
        self.hf_readme_txt.delete("1.0", tk.END)
        self.hf_readme_txt.configure(state=tk.DISABLED)

    def _hf_load_files_thread(self, request_id: int, user: str, repo: str,
                              translate: bool, requested_file: str = None,
                              revision: str = "main") -> None:
        try:
            info = self.hf.model_info(user, repo, revision)
            files = info.get("siblings", [])
            ggufs = [s for s in files if s["rfilename"].endswith(".gguf")]
            gguf_meta = {}
            if ggufs:
                try:
                    expanded_url = (
                        f"{HF_API}/models/{urllib.parse.quote(user, safe='')}/"
                        f"{urllib.parse.quote(repo, safe='')}?expand%5B%5D=gguf"
                    )
                    gguf_meta = _hf_fetch_json(expanded_url).get("gguf", {}) or {}
                except Exception:
                    pass
            if requested_file and requested_file.endswith(".gguf") and not any(
                item.get("rfilename") == requested_file for item in ggufs
            ):
                ggufs.insert(0, {"rfilename": requested_file})
            card = info.get("cardData", {}) or {}

            safetensors = len([s for s in files if s["rfilename"].endswith(".safetensors")])
            config = info.get("config", {}) or {}
            arch = config.get("model_type", card.get("base_model", "?"))
            ctx_len = ""
            if "max_position_embeddings" in config:
                ctx_len = str(config["max_position_embeddings"])
            elif "context_length" in config:
                ctx_len = str(config["context_length"])

            info_lines = [
                f"  {user}/{repo}",
                f"  ⬇ {info.get('downloads', 0):,}  ❤ {info.get('likes', 0):,}",
                f"  🏷 {info.get('pipeline_tag', '?')}  |  📐 {arch}",
            ]
            if ctx_len:
                info_lines.append(f"  🧠 ctx: {ctx_len}")
            if card.get("license"):
                info_lines.append(f"  ⚖ {card['license']}")
            if safetensors:
                info_lines.append(f"  🔧 {safetensors} safetensors")
            if card.get("language"):
                lang = card["language"]
                if isinstance(lang, list):
                    lang = ", ".join(lang[:5])
                info_lines.append(f"  🌐 {lang}")
            info_lines.append(f"  📦 {len(ggufs)} arquivos GGUF")

            info_text = "\n".join(info_lines)
            self._post_ui(
                self._hf_populate_files, request_id, ggufs, info_text, gguf_meta
            )

            # README — armazena raw e mostra conforme toggle
            try:
                quoted_revision = urllib.parse.quote(revision or "main", safe="")
                readme_url = f"https://huggingface.co/{user}/{repo}/raw/{quoted_revision}/README.md"
                req = urllib.request.Request(readme_url, headers=_hf_headers())
                with urllib.request.urlopen(req, timeout=10) as r:
                    readme_raw = r.read().decode("utf-8", errors="replace")
                if translate:
                    lines = self._hf_translate_readme(readme_raw)
                    translated = True
                else:
                    lines = readme_raw.splitlines(True)
                    translated = False
                self._post_ui(
                    self._hf_show_readme, request_id, lines, translated, readme_raw
                )
            except Exception:
                self._post_ui(self._hf_show_readme, request_id, None, False, None)

        except Exception as e:
            self._post_ui(self._hf_detail_error, request_id, f"Erro: {e}")

    def _hf_detail_error(self, request_id: int, message: str) -> None:
        if request_id == self._hf_detail_request:
            self.hf_info_var.set(message)

    def _hf_toggle_translate(self) -> None:
        if not self.hf_readme_raw:
            return
        self._hf_clear_readme()
        self.hf_readme_txt.configure(state=tk.NORMAL)
        self.hf_readme_txt.insert(tk.END, "  Recarregando...\n", "dim")
        self.hf_readme_txt.configure(state=tk.DISABLED)
        self._hf_render_request += 1
        request_id = self._hf_render_request
        raw = self.hf_readme_raw
        translate = self.hf_translate_var.get() and _hf_translate_ok
        detail_id = self._hf_detail_request
        threading.Thread(
            target=self._hf_render_readme_thread,
            args=(request_id, detail_id, raw, translate), daemon=True
        ).start()

    def _hf_render_readme_thread(self, request_id: int, detail_id: int,
                                 raw: str, translate: bool) -> None:
        if not raw:
            return
        if translate:
            lines = self._hf_translate_readme(raw)
            translated = True
        else:
            lines = raw.splitlines(True)
            translated = False
        self._post_ui(
            self._hf_render_done, request_id, detail_id, lines, translated
        )

    def _hf_render_done(self, request_id: int, detail_id: int,
                        lines: list, translated: bool) -> None:
        if request_id != self._hf_render_request or detail_id != self._hf_detail_request:
            return
        self.hf_readme_is_translated = translated
        self._render_readme(lines)

    def _hf_translate_readme(self, text: str) -> list:
        if not text:
            return []
        if not _hf_translate_ok:
            return text.splitlines(True)

        # Substitui blocos de código e tabelas por placeholders UUID
        import uuid as _uuid
        placeholders = {}

        def replacer(m):
            uid = _uuid.uuid4().hex[:12]
            placeholders[uid] = m.group(0)
            return f"~~{uid}~~"

        # Protege blocos de código ```...```
        text = re.sub(r"(?s)```.*?```", replacer, text)
        # Protege YAML front matter ---...---
        text = re.sub(r"(?s)^---\n.*?\n---\n", replacer, text)
        # Protege tabelas markdown (linhas que começam com |)
        text = re.sub(r"(?m)^\|.+\|\s*$", replacer, text)

        # Traduz o texto restante
        translated = self._hf_translate(text)
        if not translated:
            translated = text

        # Corrige #Title → # Title (Google Translate às vezes remove o espaço)
        translated = re.sub(r"^(#{1,6})([^#\s])", r"\1 \2", translated, flags=re.MULTILINE)

        # Reinsere os blocos originais
        for uid, original in placeholders.items():
            translated = translated.replace(f"~~{uid}~~", original)

        return translated.splitlines(True)

    def _hf_translate(self, text: str) -> str:
        if not _hf_translate_ok or not text:
            return text
        chunks = []
        current = ""
        for part in re.split(r"(\n\s*\n)", text):
            if len(current) + len(part) <= 7000:
                current += part
                continue
            if current:
                chunks.append(current)
            while len(part) > 7000:
                chunks.append(part[:7000])
                part = part[7000:]
            current = part
        if current:
            chunks.append(current)

        output = []
        for chunk in chunks:
            try:
                with _hf_translate_lock:
                    translated = _hf_translator.translate(chunk)
                output.append(translated if isinstance(translated, str) else translated.text)
            except Exception:
                output.append(chunk)
        return "".join(output)

    def _hf_show_readme(self, request_id: int, lines, translated: bool,
                        raw: str = None) -> None:
        if request_id != self._hf_detail_request:
            return
        self.hf_readme_raw = raw
        self.hf_readme_is_translated = translated
        self._render_readme(lines)

    def _render_readme(self, lines) -> None:
        self.hf_readme_txt.configure(state=tk.NORMAL)
        self.hf_readme_txt.delete("1.0", tk.END)
        if lines is None:
            self.hf_readme_txt.insert(tk.END, "  (README não disponível)\n", "dim")
            self.hf_readme_txt.configure(state=tk.DISABLED)
            return

        if self.hf_readme_is_translated:
            self.hf_readme_txt.insert(tk.END, "  🌐  traduzido para pt-BR", "dim")
            self.hf_readme_txt.insert(tk.END, "\n\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                self.hf_readme_txt.insert(tk.END, "\n")
            elif stripped.startswith("```"):
                self.hf_readme_txt.insert(tk.END, line, "dim")
            elif stripped.startswith("|"):
                self.hf_readme_txt.insert(tk.END, line, "dim")
            elif line.lstrip().startswith("# "):
                content = line.lstrip("# ").strip()
                self.hf_readme_txt.insert(tk.END, content + "\n", "h1")
            elif line.lstrip().startswith("## "):
                content = line.lstrip("# ").strip()
                self.hf_readme_txt.insert(tk.END, content + "\n", "h2")
            elif line.lstrip().startswith("### "):
                content = line.lstrip("# ").strip()
                self.hf_readme_txt.insert(tk.END, content + "\n", "bold")
            elif stripped.startswith("**") and stripped.endswith("**"):
                self.hf_readme_txt.insert(tk.END, stripped.strip("*") + "\n", "bold")
            elif stripped.startswith("• **"):
                self.hf_readme_txt.insert(tk.END, line, "")
            else:
                self.hf_readme_txt.insert(tk.END, line)

        self.hf_readme_txt.configure(state=tk.DISABLED)

    def _hf_populate_files(self, request_id: int, ggufs: list, info_text: str,
                           gguf_meta: dict = None) -> None:
        if request_id != self._hf_detail_request:
            return
        self.hf_ggufs = ggufs
        self.hf_info_var.set(info_text)
        self._hf_show_gguf_meta(gguf_meta or {})
        self.hf_files_lb.delete(0, tk.END)
        if not ggufs:
            self.hf_files_lb.insert(tk.END, "  (nenhum arquivo GGUF)")
            # Tenta resolver automaticamente um repo GGUF equivalente,
            # já que a maioria dos modelos trending/search só tem safetensors.
            if self.hf_selected and not self.hf_selected.get("_auto_resolved"):
                user = self.hf_selected["user"]
                repo = self.hf_selected["repo"]
                self.hf_status_var.set(
                    f"🔎  {user}/{repo} sem GGUF — procurando versão GGUF equivalente..."
                )
                self._auto_resolve_gguf_repo(request_id, user, repo)
            else:
                self.hf_status_var.set(
                    "❌  Sem GGUF neste repositório. Use 🔎 Resolver .gguf abaixo."
                )
            return
        for f in ggufs:
            sz = f.get("size", 0)
            sz_str = _hf_format_bytes(sz).rjust(10) if sz else " " * 10
            self.hf_files_lb.insert(
                tk.END, f"  {_hf_display_name(f['rfilename']):<55}  {sz_str}"
            )

    def _hf_show_gguf_meta(self, metadata: dict) -> None:
        if not metadata:
            self.hf_gguf_meta_var.set("Metadados GGUF não informados pelo repositório.")
            return
        architecture = str(metadata.get("architecture") or "N/D").upper()
        context = metadata.get("context_length")
        parameters = metadata.get("total")
        values = [f"Arquitetura: {architecture}"]
        if isinstance(context, (int, float)) and context > 0:
            values.append(f"Contexto: {int(context):,} tokens")
        if isinstance(parameters, (int, float)) and parameters > 0:
            values.append(f"Parâmetros: {_hf_format_params(int(parameters))}")
        self.hf_gguf_meta_var.set("   |   ".join(values))

    def _auto_resolve_gguf_repo(self, detail_id: int, user: str, repo: str) -> None:
        """Procura um repo GGUF equivalente ao repo base selecionado."""
        base_name = _hf_base_model_name(repo)
        search_term = f"{base_name} GGUF"
        threading.Thread(
            target=self._auto_resolve_thread,
            args=(detail_id, user, repo, search_term),
            daemon=True,
        ).start()

    def _auto_resolve_thread(self, detail_id: int, orig_user: str, orig_repo: str,
                             search_term: str) -> None:
        try:
            # Busca repos via API full=true (traz siblings)
            url = (f"{HF_API}/models?search="
                   f"{urllib.parse.quote(search_term)}&full=true&sort=likes&limit=100")
            data = _hf_fetch_json(url)
            # Aceita somente o mesmo modelo-base. Sobreposição parcial não é
            # suficiente: Qwen3-8B também aparece em muitos fine-tunes distintos.
            owner_tokens = re.findall(r"[a-z0-9]+", orig_user.lower())
            base_repo = _hf_base_model_name(orig_repo)
            model_tokens = re.findall(r"[a-z0-9]+", base_repo.lower())
            orig_norm = "".join(model_tokens)
            overlap = 0
            for size in range(1, min(len(owner_tokens), len(model_tokens)) + 1):
                if owner_tokens[-size:] == model_tokens[:size]:
                    overlap = size
            owner_model_norm = "".join(owner_tokens + model_tokens[overlap:])
            exact_names = {orig_norm, owner_model_norm}
            canonical_id = f"{orig_user}/{base_repo}".lower()
            candidates = []
            for r in data:
                sib = r.get("siblings", [])
                ggufs = [
                    s for s in sib
                    if s.get("rfilename", "").lower().endswith(".gguf")
                ]
                if not ggufs:
                    continue
                repo_id = r["id"]
                if "/" not in repo_id:
                    continue
                candidate_repo = _hf_base_model_name(repo_id.split("/", 1)[1])
                candidate_norm = re.sub(r"[^a-z0-9]", "", candidate_repo.lower())
                quantized_bases = set()
                for tag in r.get("tags", []):
                    if not isinstance(tag, str):
                        continue
                    prefix = "base_model:quantized:"
                    if tag.lower().startswith(prefix):
                        quantized_bases.add(tag[len(prefix):].lower())
                exact_name = candidate_norm in exact_names
                if not exact_name:
                    continue
                if quantized_bases and canonical_id not in quantized_bases:
                    continue
                candidate_user = repo_id.split("/", 1)[0]
                official_bonus = 100 if candidate_user.lower() == orig_user.lower() else 0
                trust = HF_TRUSTED.get(candidate_user, 0)
                candidates.append(
                    (official_bonus + trust * 10 + len(ggufs) / 10, repo_id, ggufs)
                )
            candidates.sort(key=lambda x: x[0], reverse=True)
            self._post_ui(self._auto_resolve_done, detail_id, orig_user, orig_repo,
                          candidates)
        except Exception as e:
            self._post_ui(self._hf_detail_error, detail_id, f"Erro auto-resolve: {e}")

    def _auto_resolve_done(self, detail_id: int, orig_user: str, orig_repo: str,
                           candidates: list) -> None:
        if detail_id != self._hf_detail_request:
            return
        if not candidates:
            self.hf_status_var.set(
                f"❌  Nenhuma conversão GGUF exata encontrada para "
                f"{orig_user}/{orig_repo}."
            )
            return
        best_score, best_repo_id, _best_ggufs = candidates[0]
        best_user, best_repo = best_repo_id.split("/", 1)
        # Atualiza o selecionado mantendo o user/repo de origem como referência
        self.hf_selected = {
            "user": best_user, "repo": best_repo, "file": None,
            "downloads": 0, "likes": 0, "revision": "main",
            "_auto_resolved": True, "_orig_user": orig_user, "_orig_repo": orig_repo,
        }
        # Mostra placeholder enquanto busca tamanhos via blobs=true
        self.hf_ggufs = []
        self.hf_files_lb.delete(0, tk.END)
        self.hf_files_lb.insert(tk.END, "  Carregando tamanhos...")
        self.hf_info_var.set(
            f"  Modelo original:  {orig_user}/{orig_repo}\n"
            f"  Repositório GGUF: {best_user}/{best_repo}"
        )
        self.hf_status_var.set(
            f"⏬  Carregando arquivos de {best_repo_id}..."
        )
        # Busca metadados completos (com blobs=true) para obter os tamanhos
        threading.Thread(
            target=self._auto_resolve_sizes_thread,
            args=(detail_id, best_user, best_repo, best_repo_id),
            daemon=True,
        ).start()

    def _auto_resolve_sizes_thread(self, detail_id: int, user: str, repo: str,
                                    repo_id: str) -> None:
        try:
            info = self.hf.model_info(user, repo, "main")
            sib = info.get("siblings", [])
            ggufs = [s for s in sib if s.get("rfilename", "").endswith(".gguf")
                     and s.get("size")]
            if not ggufs:
                # fallback: sem tamanhos disponíveis
                ggufs = [{"rfilename": s["rfilename"]}
                         for s in sib if s["rfilename"].endswith(".gguf")]
            gguf_meta = {}
            try:
                expanded_url = (
                    f"{HF_API}/models/{urllib.parse.quote(user, safe='')}/"
                    f"{urllib.parse.quote(repo, safe='')}?expand%5B%5D=gguf"
                )
                gguf_meta = _hf_fetch_json(expanded_url).get("gguf", {}) or {}
            except Exception:
                pass
            self._post_ui(
                self._auto_resolve_sizes_done, detail_id, ggufs, repo_id, gguf_meta
            )
        except Exception as e:
            self._post_ui(self._hf_detail_error, detail_id, f"Erro tamanhos: {e}")

    def _auto_resolve_sizes_done(self, detail_id: int, ggufs: list, repo_id: str,
                                  gguf_meta: dict) -> None:
        if detail_id != self._hf_detail_request:
            return
        self.hf_ggufs = ggufs
        self._hf_show_gguf_meta(gguf_meta)
        self.hf_files_lb.delete(0, tk.END)
        for f in ggufs:
            sz = f.get("size", 0)
            sz_str = _hf_format_bytes(sz).rjust(10) if sz else " " * 10
            self.hf_files_lb.insert(
                tk.END, f"  {_hf_display_name(f['rfilename']):<55}  {sz_str}"
            )
        self.hf_status_var.set(
            f"✔  {len(ggufs)} arquivo(s) GGUF em {repo_id}"
        )

    def _hf_on_file_select(self, _e=None) -> None:
        sel = self.hf_files_lb.curselection()
        if not sel or not self.hf_selected or sel[0] >= len(self.hf_ggufs):
            return
        fname = self.hf_ggufs[sel[0]]["rfilename"]
        self.hf_selected["file"] = fname
        self.hf_status_var.set(f"📄  Selecionado: {_hf_display_name(fname)}")

    # ── HF Download ────────────────────────────────────────
    def _hf_download(self) -> None:
        if self._download_active:
            messagebox.showwarning("Download", "Já existe um download em andamento.")
            return
        if not self.hf_selected or not self.hf_selected.get("file"):
            messagebox.showwarning("Aviso", "Selecione um arquivo GGUF na lista à direita.")
            return
        dest = self.models_dir_var.get()
        if not os.path.isdir(dest):
            mb = messagebox.askyesno(
                "Criar diretório?",
                f"O diretório não existe:\n{dest}\n\nCriar?"
            )
            if mb:
                os.makedirs(dest, exist_ok=True)
            else:
                return
        user = self.hf_selected["user"]
        repo = self.hf_selected["repo"]
        revision = self.hf_selected.get("revision", "main")
        fname = self.hf_selected["file"]
        shard = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", fname, re.I)
        if shard:
            total_shards = int(shard.group(3))
            names = [f"{shard.group(1)}-{index:05d}-of-{total_shards:05d}.gguf"
                     for index in range(1, total_shards + 1)]
            file_map = {item.get("rfilename"): item for item in self.hf_ggufs}
            missing = [name for name in names if name not in file_map]
            if missing:
                messagebox.showerror(
                    "Modelo multipartes incompleto",
                    f"O repositório não informou todas as {total_shards} partes necessárias.",
                )
                return
            if not messagebox.askyesno(
                "Modelo multipartes",
                f"Este modelo possui {total_shards} partes. Baixar todas agora?",
            ):
                return
            download_items = [file_map[name] for name in names]
        else:
            download_items = [next(
                (item for item in self.hf_ggufs if item.get("rfilename") == fname),
                {"rfilename": fname},
            )]

        existing = [str(Path(dest, item["rfilename"])) for item in download_items
                    if Path(dest, item["rfilename"]).exists()]
        if existing and not messagebox.askyesno(
            "Sobrescrever arquivos?",
            f"{len(existing)} arquivo(s) já existem e serão substituídos. Continuar?",
        ):
            return
        self._download_active = True
        self.hf_cancel_btn.configure(state=tk.NORMAL)
        self.hf_status_var.set(f"⬇️  Baixando {len(download_items)} arquivo(s)...")
        self.hf_prog_var.set("Iniciando download...")
        self._download_cancel.clear()
        self._download_thread = threading.Thread(
            target=self._hf_download_thread,
            args=(user, repo, revision, download_items, dest),
            daemon=True
        )
        self._download_thread.start()

    def _hf_cancel_download(self) -> None:
        if not self._download_active:
            return
        self._download_cancel.set()
        self.hf.cancel_downloads()
        self.hf_status_var.set("Cancelando download...")
        self.hf_cancel_btn.configure(state=tk.DISABLED)

    def _hf_download_thread(self, user: str, repo: str, revision: str,
                            items: list, dest: str) -> None:
        start = _time.monotonic()
        staged = []
        try:
            for index, item in enumerate(items, 1):
                fname = item["rfilename"]
                expected_size = int(item.get("size", 0) or 0)
                lfs_info = item.get("lfs", {}) or {}
                expected_sha256 = lfs_info.get("sha256") or lfs_info.get("oid", "")
                if expected_sha256.startswith("sha256:"):
                    expected_sha256 = expected_sha256.split(":", 1)[1]

                def prog(downloaded, total, speed, part=index, name=fname):
                    pct = downloaded / total * 100 if total else 0
                    eta = (total - downloaded) / speed if speed > 0 and total else 0
                    self._post_ui(
                        self.hf_prog_var.set,
                        f"Parte {part}/{len(items)}: {os.path.basename(name)}  "
                        f"{_hf_format_bytes(downloaded)} / {_hf_format_bytes(total)}  "
                        f"({pct:.1f}%)  {_hf_format_bytes(int(speed))}/s  ETA {eta:.0f}s",
                    )

                partial = self.hf.download(
                    user, repo, fname, dest, on_progress=prog,
                    expected_size=expected_size, expected_sha256=expected_sha256,
                    cancel_event=self._download_cancel,
                    revision=revision,
                    promote=False,
                )
                target = str(Path(partial).with_name(Path(partial).name.removesuffix(".part")))
                staged.append((partial, target))
            backups = []
            promoted = []
            try:
                for partial, target in staged:
                    backup = None
                    if os.path.exists(target):
                        backup = f"{target}.backup-{threading.get_ident()}"
                        os.replace(target, backup)
                    backups.append((target, backup))
                    os.replace(partial, target)
                    promoted.append(target)
            except Exception:
                for target in promoted:
                    try:
                        os.unlink(target)
                    except OSError:
                        pass
                for target, backup in reversed(backups):
                    if backup and os.path.exists(backup):
                        os.replace(backup, target)
                raise
            for _, backup in backups:
                if backup:
                    try:
                        os.unlink(backup)
                    except OSError:
                        pass
            paths = [target for _, target in staged]
            elapsed = _time.monotonic() - start
            self._post_ui(self._hf_download_done, paths, elapsed)
        except Exception as e:
            for partial, _ in staged:
                try:
                    os.unlink(partial)
                except OSError:
                    pass
            self._post_ui(self._hf_download_failed, str(e))

    def _hf_download_failed(self, error: str) -> None:
        self._download_active = False
        self.hf_cancel_btn.configure(state=tk.DISABLED)
        self.hf_prog_var.set("")
        if "cancel" in error.lower():
            self.hf_status_var.set("Download cancelado.")
        else:
            self.hf_status_var.set(f"❌  Erro: {error}")

    def _hf_download_done(self, paths: list, elapsed: float) -> None:
        self._download_active = False
        self.hf_cancel_btn.configure(state=tk.DISABLED)
        self.hf_prog_var.set("")
        total_size = sum(os.path.getsize(path) for path in paths)
        self.hf_status_var.set(f"✔  {len(paths)} arquivo(s) baixado(s) em {elapsed:.0f}s")
        mb = messagebox.askyesno(
            "Download concluído",
            f"{len(paths)} arquivo(s)\n\n{_hf_format_bytes(total_size)} em {elapsed:.0f}s\n\n"
            "Carregar na lista de modelos?"
        )
        if mb:
            self._load_models()
            self.nb.select(self.tab_model)

    # ── HF Resolve ─────────────────────────────────────────
    def _hf_resolve_file(self) -> None:
        fname = tk.simpledialog.askstring(
            "Resolver arquivo", "Nome do arquivo .gguf:",
            parent=self.root
        )
        if not fname:
            return
        if not fname.endswith(".gguf"):
            fname += ".gguf"
        self.hf_status_var.set(f"🔍  Resolvendo \"{fname}\"...")
        self._resolve_request += 1
        request_id = self._resolve_request
        threading.Thread(
            target=self._hf_resolve_thread, args=(request_id, fname), daemon=True
        ).start()

    def _hf_resolve_thread(self, request_id: int, fname: str) -> None:
        try:
            candidates = self.hf.resolve_candidates(fname)
            self._post_ui(self._hf_resolve_done, request_id, fname, candidates)
        except Exception as e:
            self._post_ui(self._hf_resolve_error, request_id, str(e))

    def _hf_resolve_error(self, request_id: int, error: str) -> None:
        if request_id == self._resolve_request:
            self.hf_status_var.set(f"❌  Erro: {error}")

    def _hf_resolve_done(self, request_id: int, fname: str, candidates: list) -> None:
        if request_id != self._resolve_request:
            return
        if not candidates:
            self.hf_status_var.set(f"❌  Nenhum repo encontrado para \"{fname}\"")
            messagebox.showwarning("Resolução", f"Nenhum repo encontrado para:\n{fname}")
            return
        # Pega o melhor (maior score de confiança)
        best = candidates[0]
        self.hf_search_var.set(f"{best[0]}/{best[1]}")
        self.hf_status_var.set(
            f"✔  Resolvido: {best[0]}/{best[1]}  "
            f"(+{len(candidates) - 1} candidatos)"
        )
        messagebox.showinfo(
            "Resolvido",
            f"Arquivo: {fname}\n"
            f"Repo: {best[0]}/{best[1]}\n"
            f"Trust score: {HF_TRUSTED.get(best[0], 0)}/3\n\n"
            f"A busca foi preenchida no campo acima — clique 🔍 para ver os arquivos."
        )

    # ════════════════════════════════════════════════════════
    #   TAB: PARÂMETROS
    # ════════════════════════════════════════════════════════
    def _build_tab_params(self) -> None:
        f = self.tab_params

        # Cabeçalho
        top = tk.Frame(f, bg=BG)
        top.pack(fill=tk.X, padx=16, pady=10)
        tk.Label(top, text="🔧  Parâmetros de Lançamento", bg=BG, fg=BLUE, font=FB).pack(side=tk.LEFT)
        ttk.Button(top, text="↺  Resetar para Ótimos", style="Blue.TButton",
                   command=lambda: self._fill_params(apply_profile=False)).pack(side=tk.RIGHT)
        ttk.Button(top, text="💾  Salvar Perfil JSON", style="Gray.TButton",
                   command=self._save_profile).pack(side=tk.RIGHT, padx=6)

        # Barra de info do modelo
        self.p_model_info_var = tk.StringVar(value="Nenhum modelo selecionado.")
        tk.Label(f, textvariable=self.p_model_info_var,
                 bg=BG3, fg=FG2, font=FS, padx=16, pady=5, anchor="w"
                 ).pack(fill=tk.X)

        # Área rolável
        cvs = tk.Canvas(f, bg=BG, highlightthickness=0)
        sb  = ttk.Scrollbar(f, orient=tk.VERTICAL, command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        cvs.pack(fill=tk.BOTH, expand=True)

        inner = ttk.Frame(cvs)
        win   = cvs.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.bind("<Configure>",   lambda e: cvs.itemconfig(win, width=e.width))
        for seq in ("<MouseWheel>","<Button-4>","<Button-5>"):
            cvs.bind(seq, lambda e: cvs.yview_scroll(
                -1 * (e.delta // 120 or (-1 if e.num == 4 else 1)), "units"))

        # Duas colunas de parâmetros
        cols = ttk.Frame(inner)
        cols.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        L = ttk.Frame(cols);  L.grid(row=0, column=0, sticky="nw", padx=(0,24))
        R = ttk.Frame(cols);  R.grid(row=0, column=1, sticky="nw")
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        V = self._pvars  # referência

        def addp(parent, row, key, label, choices=None, tip="", w=16,
                 memory=False):
            tk.Label(parent, text=label, bg=BG, fg=FG2, font=FS,
                     width=28, anchor="w").grid(row=row, column=0, sticky="w", pady=4, padx=2)
            var = tk.StringVar()
            V[key] = var
            if choices:
                wgt = ttk.Combobox(parent, textvariable=var, values=choices,
                                   width=w, state="readonly", font=FM)
            else:
                wgt = ttk.Entry(parent, textvariable=var, width=w+4, font=FM)
            wgt.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            if memory:
                var.trace_add(
                    "write", lambda *_args, changed=key:
                    self._schedule_memory_recalc(changed)
                )
            if tip:
                tk.Label(parent, text=tip, bg=BG, fg=FG2, font=FS
                         ).grid(row=row, column=2, sticky="w", padx=4)
            return var

        def sec(parent, row, title):
            tk.Label(parent, text=title, bg=BG, fg=CYAN, font=FS
                     ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10,2))

        # ── Coluna Esquerda ───────────────────────────────
        r = 0
        sec(L, r, "── Contexto e GPU ──────────────────────"); r+=1
        addp(L,r,"ctx",     "ctx-size (tokens)", memory=True); r+=1
        addp(L,r,"ngl",     "n-gpu-layers",  tip="999 = tudo na GPU"); r+=1
        addp(L,r,"parallel","parallel (slots)", memory=True);   r+=1
        addp(L,r,"device",  "device",        tip="Ex: CUDA0"); r+=1

        sec(L, r, "── Cache KV ─────────────────────────────"); r+=1
        KV_TYPES = ["f32","f16","bf16","q8_0","q5_0","q4_0","q4_1","iq4_nl","q5_1"]
        addp(L,r,"cache_k",   "cache-type-k",  choices=KV_TYPES,
             memory=True); r+=1
        addp(L,r,"cache_v",   "cache-type-v",  choices=KV_TYPES,
             memory=True); r+=1
        addp(L,r,"kv_unified","kv-unified",     choices=["y","n"],
             tip="Compartilha um KV entre slots");              r+=1
        addp(L,r,"kv_offload","kv-offload",     choices=["y","n"],
             tip="GPU gerencia cache KV", memory=True);          r+=1
        addp(L,r,"cache_reuse","cache-reuse",   tip="0=off, N=chunk mínimo em tokens"); r+=1
        addp(L,r,"cache_ram", "cache-ram (MiB)", memory=True); r+=1
        addp(L,r,"ctx_checkpoints", "ctx-checkpoints", memory=True); r+=1
        addp(L,r,"checkpoint_min_step", "checkpoint-min-step"); r+=1
        addp(L,r,"context_shift", "context-shift", choices=["n","y"]); r+=1

        sec(L, r, "── Atenção ──────────────────────────────"); r+=1
        addp(L,r,"flash",      "flash-attn",   choices=["auto","y","n"],
             tip="auto recomendado");                            r+=1
        addp(L,r,"split_mode", "split-mode",
             choices=["layer","none","row","tensor"]);           r+=1
        addp(L,r,"swa_full",   "swa-full",     choices=["n","y"],
             tip="Full SWA (Gemma)", memory=True);               r+=1

        sec(L, r, "── Amostragem ───────────────────────────"); r+=1
        addp(L,r,"temp",           "temperature", tip="0.60 otimizado"); r+=1
        addp(L,r,"top_k",          "top-k",       tip="20 otimizado, 0=off"); r+=1
        addp(L,r,"top_p",          "top-p",       tip="0.95 default, 1.0=off"); r+=1
        addp(L,r,"repeat_penalty", "repeat-penalty", tip="1.00 default, 1.0=off"); r+=1
        addp(L,r,"min_p",          "min-p",          tip="0.05 default, 0=off"); r+=1
        addp(L,r,"presence_penalty","presence-penalty", tip="0.0 default"); r+=1
        addp(L,r,"frequency_penalty","frequency-penalty", tip="0.0 default"); r+=1
        addp(L,r,"repeat_last_n", "repeat-last-n", tip="64 default"); r+=1
        addp(L,r,"seed", "seed", tip="-1 aleatório; 42 benchmark"); r+=1
        addp(L,r,"sampler_seq", "sampler-seq", tip="edskypmxt"); r+=1

        # ── Coluna Direita ────────────────────────────────
        r = 0
        sec(R, r, "── CPU / Batch ──────────────────────────"); r+=1
        addp(R,r,"threads",      "threads CPU");                  r+=1
        addp(R,r,"threads_batch","threads-batch", tip="0=mesmo que threads"); r+=1
        addp(R,r,"batch",        "batch-size", memory=True);      r+=1
        addp(R,r,"ubatch",       "ubatch-size (micro-batch)",
             memory=True);                                           r+=1
        addp(R,r,"poll",         "poll (0-100)", tip="100=CPU, 50=GPU"); r+=1
        addp(R,r,"numa",         "numa", choices=["none","distribute","isolate","numactl"],
             tip="Otimização NUMA");                              r+=1

        sec(R, r, "── GPU / MoE ────────────────────────────"); r+=1
        addp(R,r,"repack",    "repack",     choices=["y","n"],
             tip="Reorganiza pesos p/ GPU");                     r+=1
        addp(R,r,"cpu_moe",   "cpu-moe",    choices=["n","y"],
             tip="MoE weights na CPU");                           r+=1
        addp(R,r,"n_cpu_moe", "n-cpu-moe",  tip="0=all, N=primeiras N layers"); r+=1
        addp(R,r,"n_cpu_ffn", "n-cpu-ffn", tip="Somente modelos densos"); r+=1
        addp(R,r,"load_mode", "load-mode",
             choices=["mmap","mmap+mlock","none","mlock","dio"],
             tip="none mantém pesos CPU residentes"); r+=1
        addp(R,r,"tensor_read_lazy", "tensor-read-lazy",
             choices=["auto","on","off"], tip="Novo upstream; requer mmap"); r+=1
        addp(R,r,"no_host",   "no-host",    choices=["n","y"],
             tip="Bypass host buffer");                           r+=1
        addp(R,r,"direct_io", "direct-io",  choices=["n","y"],
             tip="DirectIO se disponível");                       r+=1

        sec(R, r, "── Rede ─────────────────────────────────"); r+=1
        addp(R,r,"host","host",  choices=["127.0.0.1","0.0.0.0"]); r+=1
        addp(R,r,"port","porta");                                 r+=1

        sec(R, r, "── Extras ───────────────────────────────"); r+=1
        addp(R,r,"mlock",  "mlock",          choices=["n","y"],
             tip="Trava modelo na RAM");                         r+=1
        addp(R,r,"no_mmap","no-mmap",         choices=["n","y"],
             tip="Evita mapeamento de memória");                 r+=1
        addp(R,r,"fit",    "Auto-fit",       choices=["n","y"],
             tip="Planeja VRAM/contexto", memory=True);             r+=1
        addp(R,r,"fit_target", "fit-target (MiB)", memory=True); r+=1
        addp(R,r,"fit_ctx", "fit-ctx mínimo"); r+=1
        addp(R,r,"jinja",  "jinja engine",   choices=["y","n"],
             tip="Template Jinja para chat");                     r+=1
        addp(R,r,"sleep_idle","sleep-idle (s)",tip="-1=off, N=segundos"); r+=1
        addp(R,r,"omni", "Omni/Visão", choices=["n","y"],
              tip="Ativa mmproj se disponível", memory=True); r+=1
        addp(R,r,"mmproj_offload","mmproj-offload",choices=["y","n"],
             tip="Projetor na GPU");                             r+=1
        addp(R,r,"mtmd_batch_max","mtmd-batch-max",tip="Max tokens img/lote"); r+=1
        addp(R,r,"image_min_tokens","image-min-tokens",tip="0=auto"); r+=1
        addp(R,r,"image_max_tokens","image-max-tokens",tip="0=auto"); r+=1
        addp(R,r,"reasoning",       "reasoning",    choices=["on","off","auto"],
             tip="Chain-of-Thought / Raciocínio");                  r+=1
        addp(R,r,"reasoning_format","reasoning-format",
             choices=["auto","deepseek","deepseek-legacy","none"],
             tip="Formato do reasoning na resposta");               r+=1
        addp(R,r,"reasoning_budget","Reasoning Budget",tip="-1=off"); r+=1
        addp(R,r,"reasoning_budget_message", "budget-message",
             tip="Mensagem quando o budget termina"); r+=1
        addp(R,r,"reasoning_preserve","reasoning-preserve",choices=["auto","y","n"],
             tip="Preserva reasoning no histórico");                r+=1
        addp(R,r,"chat_template_kwargs", "chat-template-kwargs",
             tip="JSON; ex: reasoning_effort"); r+=1
        addp(R,r,"audio",  "Áudio/Vocoder",  choices=["n","y"],
             tip="Ativa vocoder se disponível");                 r+=1
        addp(R,r,"slot_similarity","slot-similarity",tip="0=disabled, 0.10=default"); r+=1
        addp(R,r,"tools",   "Ferramentas",   choices=["disabled","readonly","all"],
             tip="all = agente completo (somente local)");       r+=1
        addp(R,r,"agentic_max_turns","agentic-max-turns",
             tip="Max ciclos de ferramentas (10 padrao, 999 = praticamente ilimitado)"); r+=1
        addp(R,r,"agentic_max_tool_preview_lines","agentic-max-tool-preview-lines",
             tip="Max linhas no preview do tool output (25 padrao)"); r+=1

        sec(R, r, "── Speculative Decoding ────────────────────"); r+=1
        SPEC_TYPES = ["none","ngram-mod","draft-mtp","draft-simple","draft-eagle3","draft-dflash","ngram-simple","ngram-map-k","ngram-map-k4v","ngram-cache"]
        addp(R,r,"spec_type",       "spec-type",       choices=SPEC_TYPES,
             tip="draft-mtp = self-spec (MTP heads no modelo)", memory=True); r+=1
        addp(R,r,"spec_draft_n_max","spec-draft-n-max",
             tip="Max draft tokens (3 p/ MTP, 8 p/ ngram)"); r+=1
        addp(R,r,"spec_draft_n_min","spec-draft-n-min", tip="0=auto"); r+=1
        addp(R,r,"spec_draft_p_min","spec-draft-p-min",
             tip="Probabilidade minima (0.0=greedy, 0.5=conservador)"); r+=1
        addp(R,r,"spec_draft_p_split","spec-draft-p-split", tip="0.10 default"); r+=1

        sec(R, r, "── Servidor ───────────────────────────────"); r+=1
        addp(R,r,"warmup", "warmup", choices=["y","n"]); r+=1
        addp(R,r,"timeout", "timeout HTTP (s)"); r+=1
        addp(R,r,"log_verbosity", "log verbosity", choices=["0","1","2","3","4","5"]); r+=1
        addp(R,r,"metrics", "métricas", choices=["n","y"]); r+=1
        addp(R,r,"backend_sampling", "backend-sampling",
             choices=["auto","y","n"], tip="CUDA quando suportado"); r+=1
        addp(R,r,"cont_batching", "continuous batching", choices=["y","n"]); r+=1
        addp(R,r,"cache_prompt", "cache-prompt", choices=["y","n"]); r+=1
        addp(R,r,"cache_idle_slots", "cache-idle-slots", choices=["y","n"]); r+=1
        addp(R,r,"offline", "offline", choices=["y","n"],
             tip="Bloqueia downloads do servidor"); r+=1
        addp(R,r,"alias", "model alias", tip="ID em /v1/models"); r+=1

        # Painel de justificativas
        jf = tk.Frame(inner, bg=BG2, padx=14, pady=10)
        jf.pack(fill=tk.X, padx=16, pady=8)
        tk.Label(jf, text="Justificativas dos valores ótimos:", bg=BG2, fg=FG2, font=FS
                 ).pack(anchor="w")
        self.p_reason_var = tk.StringVar()
        tk.Label(jf, textvariable=self.p_reason_var,
                 bg=BG2, fg=FG2, font=FS, justify="left",
                 wraplength=1000, pady=4, anchor="w"
                 ).pack(fill=tk.X)

        # Botão próxima aba
        bf = ttk.Frame(inner)
        bf.pack(fill=tk.X, padx=16, pady=12)
        ttk.Button(bf, text="▶   Ir para Lançar →",
                   style="Green.TButton",
                   command=self._go_launch).pack(side=tk.RIGHT)

    def _fill_params(self, apply_profile: bool = True) -> None:
        if self.opt is None:
            return
        self._updating_params = True
        o = self.opt; v = self._pvars

        def sv(k, val):
            if k in v: v[k].set(str(val))

        sv("ctx",               o.ctx)
        sv("ngl",               o.ngl)
        sv("parallel",          o.parallel)
        sv("cache_k",           o.cache_k)
        sv("cache_v",           o.cache_v)
        sv("kv_unified",        "y" if o.kv_unified else "n")
        sv("kv_offload",        o.kv_offload)
        sv("flash",             o.flash)
        sv("split_mode",        o.split_mode)
        sv("device",            o.device)
        sv("numa",              o.numa)
        sv("repack",            o.repack)
        sv("direct_io",         o.direct_io)
        sv("no_host",           o.no_host)
        sv("load_mode",         o.load_mode)
        sv("tensor_read_lazy",  o.tensor_read_lazy)
        sv("swa_full",          o.swa_full)
        sv("cache_reuse",       o.cache_reuse)
        sv("threads",           o.threads)
        sv("threads_batch",     o.threads_batch)
        sv("batch",             o.batch)
        sv("ubatch",            o.ubatch)
        sv("poll",              o.poll)
        sv("host",              o.host)
        sv("port",              o.port)
        sv("mlock",             o.mlock)
        sv("no_mmap",           "n")
        sv("temp",              o.temp)
        sv("top_k",             o.top_k)
        sv("top_p",             o.top_p)
        sv("repeat_penalty",    o.repeat_penalty)
        sv("min_p",             o.min_p)
        sv("presence_penalty",  o.presence_penalty)
        sv("frequency_penalty", o.frequency_penalty)
        sv("repeat_last_n",     o.repeat_last_n)
        sv("seed",              o.seed)
        sv("sampler_seq",       o.sampler_seq)
        sv("reasoning",         o.reasoning)
        sv("reasoning_format",  o.reasoning_format)
        sv("reasoning_budget",  o.reasoning_budget)
        sv("reasoning_budget_message", o.reasoning_budget_message)
        sv("reasoning_preserve", o.reasoning_preserve)
        sv("chat_template_kwargs", o.chat_template_kwargs)
        sv("omni",              o.omni)
        sv("mmproj_offload",    o.mmproj_offload)
        sv("mtmd_batch_max",    o.mtmd_batch_max)
        sv("audio",             o.audio)
        sv("image_min_tokens",  o.image_min_tokens)
        sv("image_max_tokens",  o.image_max_tokens)
        sv("cpu_moe",           o.cpu_moe)
        sv("n_cpu_moe",         o.n_cpu_moe)
        sv("n_cpu_ffn",         o.n_cpu_ffn)
        sv("sleep_idle",        o.sleep_idle)
        sv("jinja",             o.jinja)
        sv("slot_similarity",   o.slot_similarity)
        sv("fit",               o.fit)
        sv("fit_target",        o.fit_target)
        sv("fit_ctx",           min(o.fit_ctx, o.ctx))
        sv("cache_ram",         o.cache_ram)
        sv("ctx_checkpoints",   o.ctx_checkpoints)
        sv("checkpoint_min_step", o.checkpoint_min_step)
        sv("context_shift",     o.context_shift)
        sv("warmup",            o.warmup)
        sv("timeout",           o.timeout)
        sv("log_verbosity",     o.log_verbosity)
        sv("metrics",           o.metrics)
        sv("tools",             "all")
        sv("agentic_max_turns", o.agentic_max_turns)
        sv("agentic_max_tool_preview_lines", o.agentic_max_tool_preview_lines)
        sv("spec_type",         o.spec_type)
        sv("spec_draft_n_max",  o.spec_draft_n_max)
        sv("spec_draft_n_min",  o.spec_draft_n_min)
        sv("spec_draft_p_min",  o.spec_draft_p_min)
        sv("spec_draft_p_split", o.spec_draft_p_split)
        sv("backend_sampling",  o.backend_sampling)
        sv("cont_batching",     o.cont_batching)
        sv("cache_prompt",      o.cache_prompt)
        sv("cache_idle_slots",  o.cache_idle_slots)
        sv("offline",           o.offline)
        sv("alias",             o.alias or Path(self.model_path).stem)

        mt = self.meta
        if mt:
            profile_params = dict(mt.profile_parameters)
            if "spec_mode" in profile_params and "spec_type" not in profile_params:
                profile_params["spec_type"] = "ngram-mod" if profile_params["spec_mode"] == "ngram" else "none"
            if apply_profile:
                for key, value in profile_params.items():
                    if key in v:
                        v[key].set(str(value))
            self.p_model_info_var.set(
                f"  Modelo: {os.path.basename(self.model_path)}  |  "
                f"{mt.size_gb_str} GB  |  quant: {mt.quant}  |  arch: {mt.arch}  |  "
                f"ctx: {mt.ctx_max}  |  layers: {mt.layers}  |  "
                f"heads: {mt.heads}/{mt.heads_kv}  |  head_dim: {mt.head_dim}"
                f"{'  |  PERFIL JSON ATIVO' if mt.profile_file else ''}"
            )
            reasons = (
                f"contexto ........ {o.ctx} tokens ({o.ctx_reason})\n"
                f"ngl ............. {o.ngl_reason}\n"
                f"cache K/V ....... {o.cache_reason}\n"
                f"kv-unified ...... {o.kv_reason}\n"
                f"kv-offload ...... {o.kv_offload_reason}\n"
                f"flash-attn ...... {o.flash_reason}\n"
                f"threads ........ {o.threads_reason}\n"
                f"batch .......... {o.batch_reason}\n"
                f"numa ........... {o.numa_reason}\n"
                f"device ......... {o.device_reason}\n"
                f"omni/visao ..... {o.omni_reason}\n"
                f"mmproj-offload . {o.mmproj_offload_reason}\n"
                f"mtmd-batch ..... {o.mtmd_batch_reason}\n"
                f"image-min-tok .. {o.image_min_tokens_reason}\n"
                f"swa-full ....... {o.swa_reason}\n"
                f"cache-reuse .... {o.cache_reuse_reason}\n"
                f"reasoning ...... {o.reasoning} / {o.reasoning_format}\n"
                f"reasoning-preserve ....... {o.reasoning_preserve} ({o.reasoning_preserve_reason})\n"
                f"speculative ...... {o.spec_type} (n_max={o.spec_draft_n_max}, p_min={o.spec_draft_p_min})"
            )
            if mt.profile_file:
                reasons += f"\nperfil JSON ..... {mt.profile.get('name', os.path.basename(mt.profile_file))}"
            self.p_reason_var.set(reasons)
        self._updating_params = False

    def _schedule_memory_recalc(self, changed: str, delay: int = 350) -> None:
        """Debounce expensive fit calculations without blocking Tk's event loop."""
        if self._updating_params or not self.meta or not self.opt:
            return
        if self.proc and self.proc.poll() is None:
            return
        self._recalc_request += 1
        request_id = self._recalc_request
        if self._recalc_after is not None:
            try:
                self.root.after_cancel(self._recalc_after)
            except (tk.TclError, ValueError):
                pass
        self.status_var.set(f"Calculando VRAM/contexto após alterar {changed}...")
        self._recalc_after = self.root.after(
            max(int(delay), 0), self._start_memory_recalc, request_id, changed
        )

    def _start_memory_recalc(self, request_id: int, changed: str) -> None:
        if request_id != self._recalc_request or not self.meta:
            return
        self._recalc_after = None

        def value(name, default):
            var = self._pvars.get(name)
            return var.get().strip() if var is not None else str(default)

        try:
            snapshot = {
                "ctx": max(int(value("ctx", self.opt.ctx)), 1),
                "parallel": max(int(value("parallel", 1)), 1),
                "fit_target": max(int(value("fit_target", self.opt.fit_target)), 0),
                "fit_ctx": max(int(value("fit_ctx", self.opt.fit_ctx)), 1),
                "fit": value("fit", "y"),
                "cache_k": value("cache_k", self.opt.cache_k).lower(),
                "cache_v": value("cache_v", self.opt.cache_v).lower(),
                "kv_offload": value("kv_offload", self.opt.kv_offload).lower(),
                "batch": max(int(value("batch", self.opt.batch)), 1),
                "ubatch": max(int(value("ubatch", self.opt.ubatch)), 1),
                "cache_ram": max(int(value("cache_ram", self.opt.cache_ram)), 0),
                "ctx_checkpoints": max(int(
                    value("ctx_checkpoints", self.opt.ctx_checkpoints)
                ), 0),
                "spec_type": value("spec_type", self.opt.spec_type).lower(),
                "swa_full": value("swa_full", self.opt.swa_full).lower(),
                "omni": value("omni", self.opt.omni).lower(),
            }
        except (TypeError, ValueError):
            self.status_var.set("⚠  Aguarde: um campo de memória ainda não contém um número válido.")
            return
        if _requires_symmetric_kv(self.meta) and snapshot["cache_k"] != snapshot["cache_v"]:
            if changed == "cache_v":
                snapshot["cache_k"] = snapshot["cache_v"]
            else:
                snapshot["cache_v"] = snapshot["cache_k"]
        if snapshot["ubatch"] > snapshot["batch"]:
            self.status_var.set("⚠  ubatch não pode ser maior que batch.")
            return

        model_path = self.model_path
        hardware = self.hw
        metadata = self.meta
        self._recalc_running = True
        threading.Thread(
            target=self._memory_recalc_thread,
            args=(request_id, changed, model_path, hardware, metadata, snapshot),
            daemon=True,
        ).start()

    def _memory_recalc_thread(self, request_id: int, changed: str,
                              model_path: str, hardware: HardwareInfo,
                              metadata: ModelMetadata, snapshot: dict) -> None:
        try:
            opt = OptimalParams(
                hardware, metadata, llama_server=LLAMA_SERVER,
                llama_fit_params=LLAMA_FIT_PARAMS,
            )
            opt.parallel = snapshot["parallel"]
            opt.fit_target = snapshot["fit_target"]
            opt.fit_ctx = snapshot["fit_ctx"]
            opt.fit = "y" if snapshot["fit"] == "y" else "n"
            opt.cache_k = snapshot["cache_k"]
            opt.cache_v = snapshot["cache_v"]
            opt.cache_ram = snapshot["cache_ram"]
            opt.ctx_checkpoints = snapshot["ctx_checkpoints"]
            opt.batch = snapshot["batch"]
            opt.ubatch = snapshot["ubatch"]
            opt.spec_type = snapshot["spec_type"]
            opt.swa_full = snapshot["swa_full"]
            opt.vision_enabled = snapshot["omni"] == "y"
            opt.calculate()
            opt.spec_type = snapshot["spec_type"]
            opt.swa_full = snapshot["swa_full"]
            opt.recalculate_memory(
                snapshot["cache_k"], snapshot["cache_v"],
                snapshot["batch"], snapshot["ubatch"],
            )
            # ``recalculate_memory`` seleciona o padrão automático.  Quando
            # o usuário editou o offload, ele é uma escolha explícita e o
            # plano de host/swap deve refletir exatamente essa escolha.
            if changed == "kv_offload" and snapshot["kv_offload"] in {"y", "n"}:
                opt.kv_offload = snapshot["kv_offload"]
                opt.kv_offload_reason = "selecionado pelo usuário"
            architecture = metadata.arch.lower()
            if architecture == "laguna":
                if not opt._adapt_laguna(snapshot["ctx"]):
                    raise ValueError("o contexto solicitado não cabe no plano Laguna MoE")
            elif architecture == "gemma4" and metadata.expert_count > 0:
                if not opt._adapt_gemma4_moe(snapshot["ctx"]):
                    raise ValueError("o contexto solicitado não cabe no plano Gemma 4 MoE")
            elif architecture == "qwen35moe":
                if not opt._adapt_qwen35moe(snapshot["ctx"]):
                    raise ValueError("o contexto solicitado não cabe no plano Qwen35 MoE")
            elif architecture == "qwen3moe":
                if not opt._adapt_qwen3moe(snapshot["ctx"]):
                    raise ValueError("o contexto solicitado não cabe no plano Qwen3 MoE")
            else:
                opt.ctx = min(snapshot["ctx"], metadata.ctx_max)
            # ``OptimalParams.calculate()`` recalcula o piso do Fit a partir
            # do contexto para o perfil automático. Durante uma edição na
            # GUI legada, porém, ``fit_ctx`` é um valor explícito do usuário
            # e precisa sobreviver ao recálculo de K/V, batch ou offload.
            # Restaure-o depois da adaptação de arquitetura, quando ``ctx``
            # já é o valor efetivamente planejado.
            opt.fit_ctx = min(
                max(int(snapshot["fit_ctx"]), 1),
                max(int(opt.ctx), 1),
                max(int(metadata.ctx_max), 1),
            )
            # A adaptação de arquitetura acima pode ter ajustado ``ctx``.
            # Planeje somente agora: cache de prompt e checkpoints dependem
            # da janela efetiva e precisam entrar na reserva de swap antes
            # de o llama-server ser iniciado.
            opt.cache_ram = snapshot["cache_ram"]
            opt.ctx_checkpoints = snapshot["ctx_checkpoints"]
            opt._plan_host_memory()
            self._post_ui(
                self._memory_recalc_done, request_id, changed, model_path,
                snapshot, opt, "",
            )
        except Exception as exc:
            self._post_ui(
                self._memory_recalc_done, request_id, changed, model_path,
                snapshot, None, str(exc),
            )

    def _memory_recalc_done(self, request_id: int, changed: str,
                            model_path: str, snapshot: dict,
                            opt: OptimalParams | None, error: str) -> None:
        if request_id != self._recalc_request or model_path != self.model_path:
            return
        self._recalc_running = False
        if error or opt is None:
            self.status_var.set(f"⚠  Falha ao recalcular: {error}")
            return
        self.opt = opt
        derived = {
            "ctx": opt.ctx,
            "ngl": opt.ngl,
            "cache_k": opt.cache_k,
            "cache_v": opt.cache_v,
            "flash": opt.flash,
            "cpu_moe": opt.cpu_moe,
            "n_cpu_moe": opt.n_cpu_moe,
            "kv_offload": opt.kv_offload,
            "mmproj_offload": opt.mmproj_offload,
            "mtmd_batch_max": opt.mtmd_batch_max,
            "image_min_tokens": opt.image_min_tokens,
            "image_max_tokens": opt.image_max_tokens,
            "load_mode": opt.load_mode,
            "backend_sampling": opt.backend_sampling,
            "fit_ctx": opt.fit_ctx,
        }
        self._updating_params = True
        try:
            for key, value in derived.items():
                if key in self._pvars:
                    self._pvars[key].set(str(value))
        finally:
            self._updating_params = False
        self.p_reason_var.set(
            f"contexto ........ {opt.ctx} tokens ({opt.ctx_reason})\n"
            f"Fit efetivo .... {opt.fit_plan_reason or f'margem {opt.fit_target} MiB'}\n"
            f"GPU/MoE ........ {opt.ngl_reason}; {opt.n_cpu_moe_reason}\n"
            f"cache K/V ...... {opt.cache_reason}\n"
            f"host/cache ..... {opt.host_growth_reason}\n"
            f"swap NVMe ...... {opt.swap_plan_reason}\n"
            f"sampling ....... {opt.sampling_reason}\n"
            f"load-mode ...... {opt.load_mode_reason or opt.load_mode}\n"
            f"backend ........ {opt.backend_sampling_reason}\n"
            f"visão .......... {opt.mmproj_offload_reason}"
        )
        self.status_var.set(
            f"✔  Plano recalculado ({changed}): {opt.ctx} tokens, "
            f"{opt.n_cpu_moe} camadas MoE na CPU."
        )
        self._update_cmd_preview()


    def _go_launch(self) -> None:
        self._update_cmd_preview()
        self.nb.select(self.tab_launch)

    def _save_profile(self) -> None:
        try:
            self._get_final()
        except ValueError as exc:
            messagebox.showerror("Parâmetro inválido", str(exc))
            return
        if not self.model_path:
            return
        profile_path = Path(self.model_path).with_suffix(".launch.json")
        profile = dict(self.meta.profile) if self.meta and self.meta.profile else {}
        values = {key: var.get().strip() for key, var in self._pvars.items()}
        active = profile.get("active_preset")
        presets = profile.get("presets", {})
        if active and isinstance(presets, dict) and isinstance(presets.get(active), dict):
            if not messagebox.askyesno(
                "Atualizar preset ativo?",
                f"O perfil usa o preset '{active}'. Salvar os valores atuais nesse preset?",
            ):
                return
            updated_presets = dict(presets)
            updated_preset = dict(updated_presets[active])
            updated_preset.update(values)
            updated_presets[active] = updated_preset
            profile["presets"] = updated_presets
        else:
            profile["parameters"] = values
        profile.update({
            "name": profile.get("name", profile_path.stem),
            "model_file": Path(self.model_path).name,
        })
        temp_path = profile_path.with_name(profile_path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(profile, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_path, profile_path)
        except OSError as exc:
            try:
                temp_path.unlink()
            except OSError:
                pass
            messagebox.showerror("Erro ao salvar perfil", str(exc))
            return
        if self.meta:
            self.meta.profile_file = str(profile_path)
            self.meta.profile = profile
            self.meta.profile_parameters = values
        messagebox.showinfo("Perfil salvo", f"Perfil gravado em:\n{profile_path}")

    def _get_final(self) -> dict:
        v = self._pvars
        def gi(k, d=0):
            value = v[k].get().strip() if k in v else str(d)
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{k}: informe um número inteiro válido") from None
        def gs(k, d=""):
            try: return v[k].get().strip()
            except Exception: return d
        final = {
            "model_path":       self.model_path,
            "ctx":              gi("ctx",              self.opt.ctx if self.opt else 4096),
            "ngl":              gs("ngl",              "auto"),
            "parallel":         gi("parallel",         1),
            "cache_k":          gs("cache_k",          "f16"),
            "cache_v":          gs("cache_v",          "f16"),
            "kv_unified":       gs("kv_unified",       "n") == "y",
            "kv_offload":       gs("kv_offload",       "y"),
            "flash":            gs("flash",            "auto"),
            "split_mode":       gs("split_mode",       "layer"),
            "device":           gs("device",           ""),
            "numa":             gs("numa",             "none"),
            "repack":           gs("repack",           "y"),
            "load_mode":        gs("load_mode",        "mmap"),
            "tensor_read_lazy": gs("tensor_read_lazy", "auto"),
            "direct_io":        gs("direct_io",        "n"),
            "no_host":          gs("no_host",          "n"),
            "swa_full":         gs("swa_full",         "n"),
            "cache_reuse":      gi("cache_reuse",      0),
            "threads":          gi("threads",          8),
            "threads_batch":    gi("threads_batch",    0),
            "batch":            gi("batch",            2048),
            "ubatch":           gi("ubatch",           256),
            "poll":             gi("poll",             50),
            "host":             gs("host",             "127.0.0.1"),
            "port":             gi("port",             8080),
            "mlock":            v["mlock"].get() or "n",
            "no_mmap":          v["no_mmap"].get() or "n",
            "temp":             gs("temp",              "0.80"),
            "top_k":            gi("top_k",             40),
            "top_p":            gs("top_p",             "0.95"),
            "repeat_penalty":   gs("repeat_penalty",    "1.00"),
            "min_p":            gs("min_p",             "0.05"),
            "presence_penalty": gs("presence_penalty",   "0.00"),
            "frequency_penalty": gs("frequency_penalty", "0.00"),
            "repeat_last_n":    gi("repeat_last_n",    64),
            "seed":             gi("seed",             -1),
            "sampler_seq":      gs("sampler_seq",      "edskypmxt"),
            "reasoning":        gs("reasoning",        "auto"),
            "reasoning_format": gs("reasoning_format", "auto"),
            "reasoning_budget":    gi("reasoning_budget",    -1),
            "reasoning_budget_message": gs("reasoning_budget_message", ""),
            "reasoning_preserve":  gs("reasoning_preserve", "auto"),
            "chat_template_kwargs": gs("chat_template_kwargs", ""),
            "omni":                gs("omni",                "n"),
            "mmproj_offload":   gs("mmproj_offload",   "y"),
            "mtmd_batch_max":   gi("mtmd_batch_max",   1024),
            "audio":            gs("audio",            "n"),
            "image_min_tokens": gi("image_min_tokens", 0),
            "image_max_tokens": gi("image_max_tokens", 0),
            "cpu_moe":          gs("cpu_moe",          "n"),
            "n_cpu_moe":        gi("n_cpu_moe",        0),
            "n_cpu_ffn":        gi("n_cpu_ffn",        0),
            "sleep_idle":       gi("sleep_idle",       -1),
            "jinja":            gs("jinja",            "y"),
            "slot_similarity":  gs("slot_similarity",  "0.10"),
            "media_path":       MEDIA_PATH if os.path.isdir(MEDIA_PATH) else "",
            "tools":            gs("tools",            "all"),
            "fit":              gs("fit",              "y"),
            "fit_target":       gi("fit_target",       1024),
            "fit_ctx":          gi("fit_ctx",          4096),
            "cache_ram":        gi("cache_ram",        2048),
            "ctx_checkpoints":  gi("ctx_checkpoints",  32),
            "checkpoint_min_step": gi("checkpoint_min_step", 8192),
            "context_shift":    gs("context_shift",    "n"),
            "warmup":           gs("warmup",           "y"),
            "timeout":          gi("timeout",          3600),
            "log_verbosity":    gi("log_verbosity",    3),
            "metrics":          gs("metrics",          "n"),
            "backend_sampling": gs("backend_sampling", "auto"),
            "cont_batching":    gs("cont_batching",    "y"),
            "cache_prompt":     gs("cache_prompt",     "y"),
            "cache_idle_slots": gs("cache_idle_slots", "y"),
            "offline":          gs("offline",          "n"),
            "alias":            gs("alias", Path(self.model_path).stem),
            "tags":             "",
            "agentic_max_turns": gi("agentic_max_turns", 10),
            "agentic_max_tool_preview_lines": gi("agentic_max_tool_preview_lines", 25),
            "spec_type":         gs("spec_type",         "none"),
            "spec_draft_n_max":  gi("spec_draft_n_max",  0),
            "spec_draft_n_min":  gi("spec_draft_n_min",  0),
            "spec_draft_p_min":  gs("spec_draft_p_min",  0.0),
            "spec_draft_p_split": gs("spec_draft_p_split", 0.10),
        }
        final["agentic"] = "y"
        final["mcp_config_file"] = ""
        final["mcp_config_json"] = ""
        if self.mcp_enabled.get():
            node = shutil.which("node")
            if not node:
                raise ValueError("Node.js não encontrado; necessário para o MCP local")
            if not LOCAL_MCP_ENTRY.is_file():
                raise ValueError(f"MCP local não encontrado: {LOCAL_MCP_ENTRY}")
            if not (LOCAL_MCP_DIR / "node_modules" / "@modelcontextprotocol" / "sdk").is_dir():
                raise ValueError(
                    f"dependências MCP ausentes; execute npm install em {LOCAL_MCP_DIR}"
                )
            if not LOCAL_MCP_WORKSPACE.is_dir():
                raise ValueError(f"workspace MCP local não encontrado: {LOCAL_MCP_WORKSPACE}")
            native_config = {
                "mcpServers": {
                    "crono-matrix": {
                        "command": node,
                        "args": [str(LOCAL_MCP_ENTRY)],
                        "cwd": str(LOCAL_MCP_DIR),
                        "timeout_ms": 120000,
                        "env": {
                            "MCP_STDIO": "1",
                            "CRONO_PROJECT_ROOT": str(PROJECT_ROOT),
                            "CRONO_WORKSPACE": str(LOCAL_MCP_WORKSPACE),
                            "CRONO_MODELS_DIR": self.models_dir_var.get(),
                            "CRONO_LLAMA_PATH": LLAMA_SERVER,
                            "CRONO_LLAMA_HOST": str(final["host"]),
                            "CRONO_LLAMA_PORT": str(final["port"]),
                            "CRONO_MCP_TOOL_POLICY": "safe",
                            "CRONO_SNN_THREADS": "4",
                            "CRONO_SNN_STEPS": "64",
                            "CRONO_SNN_TIMEOUT_MS": "15000",
                            "CRONO_MEMORY_DIR": str(LOCAL_MCP_MEMORY),
                            "CRONO_SNN_ENABLED_FILE": str(LOCAL_SNN_ENABLED),
                            "CRONO_MCP_REPEAT_LIMIT": "3",
                            "LOG_LEVEL": "info",
                        },
                    },
                },
            }
            final["mcp_config_json"] = json.dumps(
                native_config, ensure_ascii=True, separators=(",", ":")
            )
        return self._validate_final(final)

    def _validate_final(self, final: dict) -> dict:
        def integer(name, minimum, maximum=None):
            value = final[name]
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ValueError(f"{name}: valor inteiro inválido") from None
            if value < minimum or (maximum is not None and value > maximum):
                limit = f" entre {minimum} e {maximum}" if maximum is not None else f" >= {minimum}"
                raise ValueError(f"{name}: deve estar{limit}")
            final[name] = value

        def decimal(name, minimum, maximum=None):
            try:
                value = float(final[name])
            except (TypeError, ValueError):
                raise ValueError(f"{name}: informe um número decimal válido") from None
            if not math.isfinite(value):
                raise ValueError(f"{name}: informe um número decimal finito")
            if value < minimum or (maximum is not None and value > maximum):
                limit = f" entre {minimum} e {maximum}" if maximum is not None else f" >= {minimum}"
                raise ValueError(f"{name}: deve estar{limit}")
            final[name] = value

        ctx_max = self.meta.ctx_max if self.meta else None
        integer("ctx", 1, ctx_max)
        if str(final["ngl"]).lower() in {"auto", "all"}:
            final["ngl"] = str(final["ngl"]).lower()
        else:
            try:
                final["ngl"] = int(final["ngl"])
            except (TypeError, ValueError):
                raise ValueError("ngl: use auto, all ou um inteiro entre 0 e 999") from None
            if not 0 <= final["ngl"] <= 999:
                raise ValueError("ngl: deve estar entre 0 e 999")

        for name, minimum, maximum in (
            ("parallel", -1, 128), ("cache_reuse", 0, None),
            ("threads", 1, 1024), ("threads_batch", 0, 1024),
            ("batch", 1, None), ("ubatch", 1, None), ("poll", 0, 100),
            ("port", 1, 65535), ("top_k", 0, None),
            ("reasoning_budget", -1, None), ("mtmd_batch_max", 0, None),
            ("image_min_tokens", 0, None), ("image_max_tokens", 0, None),
            ("n_cpu_moe", 0, None), ("n_cpu_ffn", 0, None),
            ("repeat_last_n", -1, None), ("seed", -1, None),
            ("sleep_idle", -1, None),
            ("fit_target", 0, None), ("fit_ctx", 1, None),
            ("cache_ram", -1, None), ("ctx_checkpoints", 0, None),
            ("checkpoint_min_step", 0, None), ("timeout", 1, None),
            ("log_verbosity", 0, 5), ("agentic_max_turns", 0, None),
            ("agentic_max_tool_preview_lines", 0, None),
            ("spec_draft_n_max", 0, None), ("spec_draft_n_min", 0, None),
        ):
            integer(name, minimum, maximum)

        for name, minimum, maximum in (
            ("temp", 0.0, 5.0), ("top_p", 0.0, 1.0),
            ("min_p", 0.0, 1.0), ("repeat_penalty", 0.0, 10.0),
            ("presence_penalty", -2.0, 2.0), ("slot_similarity", 0.0, 1.0),
            ("spec_draft_p_min", 0.0, 1.0),
            ("frequency_penalty", -2.0, 2.0),
            ("spec_draft_p_split", 0.0, 1.0),
        ):
            decimal(name, minimum, maximum)

        allowed = {
            "cache_k": {"f32", "f16", "bf16", "q8_0", "q5_0", "q4_0", "q4_1", "iq4_nl", "q5_1"},
            "cache_v": {"f32", "f16", "bf16", "q8_0", "q5_0", "q4_0", "q4_1", "iq4_nl", "q5_1"},
            "flash": {"auto", "y", "n"},
            "split_mode": {"layer", "none", "row", "tensor"},
            "numa": {"none", "distribute", "isolate", "numactl"},
            "reasoning": {"on", "off", "auto"},
            "reasoning_format": {"auto", "deepseek", "deepseek-legacy", "none"},
            "reasoning_preserve": {"auto", "y", "n"},
            "tools": {"disabled", "readonly", "all"},
            "load_mode": {"mmap", "mmap+mlock", "mlock", "none", "dio"},
            "tensor_read_lazy": {"auto", "on", "off"},
            "backend_sampling": {"auto", "y", "n"},
            "spec_type": {"none", "ngram-mod", "draft-mtp", "draft-simple", "draft-eagle3", "draft-dflash", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-cache"},
        }
        for name, choices in allowed.items():
            if final[name] not in choices:
                raise ValueError(f"{name}: opção inválida ({final[name]})")
        for name in ("kv_offload", "repack", "direct_io", "no_host", "swa_full",
                     "mlock", "no_mmap", "omni", "mmproj_offload", "audio",
                     "cpu_moe", "jinja", "fit", "context_shift", "warmup", "metrics",
                     "cont_batching", "cache_prompt", "cache_idle_slots", "offline"):
            if final[name] not in {"y", "n"}:
                raise ValueError(f"{name}: use y ou n")
        if final["ubatch"] > final["batch"]:
            raise ValueError("ubatch deve ser menor ou igual ao batch")
        if final["parallel"] == 0:
            raise ValueError("parallel: use -1 para automático ou um valor maior que zero")
        if final["fit"] == "y" and final["fit_ctx"] > final["ctx"]:
            raise ValueError("fit_ctx deve ser menor ou igual ao contexto")
        if final["image_max_tokens"] and final["image_max_tokens"] < final["image_min_tokens"]:
            raise ValueError("image_max_tokens deve ser >= image_min_tokens")
        if not final["host"]:
            raise ValueError("host não pode ficar vazio")
        if final["tools"] != "disabled":
            try:
                loopback = final["host"].lower() == "localhost" or ipaddress.ip_address(
                    final["host"].split("%", 1)[0]
                ).is_loopback
            except ValueError:
                loopback = False
            if not loopback:
                raise ValueError("ferramentas locais exigem um host de loopback")
        if final["omni"] == "y" and not (self.meta and self.meta.mmproj_valid):
            raise ValueError("omni exige um mmproj compatível e validado")
        if final["tensor_read_lazy"] == "on" and final["load_mode"] not in {
            "mmap", "mmap+mlock",
        }:
            raise ValueError("tensor_read_lazy=on exige load_mode mmap ou mmap+mlock")
        if final["tensor_read_lazy"] != "auto" and not _server_supports_flag(
            LLAMA_SERVER, "--tensor-read-lazy"
        ):
            raise ValueError("o llama-server selecionado não suporta --tensor-read-lazy")
        if self.meta:
            if final["n_cpu_moe"] > self.meta.layers:
                raise ValueError(f"n_cpu_moe deve estar entre 0 e {self.meta.layers}")
            if final["n_cpu_ffn"] > self.meta.layers:
                raise ValueError(f"n_cpu_ffn deve estar entre 0 e {self.meta.layers}")
            if self.meta.moe_layers and final["n_cpu_ffn"] > 0:
                raise ValueError("n_cpu_ffn é exclusivo para modelos densos; use n_cpu_moe")
        if final["n_cpu_ffn"] > 0 and not _server_supports_flag(
            LLAMA_SERVER, "--n-cpu-ffn"
        ):
            raise ValueError("o llama-server selecionado não suporta --n-cpu-ffn")
        if final["chat_template_kwargs"]:
            try:
                value = json.loads(final["chat_template_kwargs"])
            except (ValueError, TypeError) as exc:
                raise ValueError("chat_template_kwargs deve ser um objeto JSON válido") from exc
            if not isinstance(value, dict):
                raise ValueError("chat_template_kwargs deve ser um objeto JSON")
        if self.meta and not self.meta.meta_ok:
            raise ValueError(
                "os metadados GGUF essenciais não foram lidos; verifique a biblioteca gguf e o arquivo"
            )
        if not os.path.isfile(final["model_path"]):
            raise ValueError("arquivo do modelo não encontrado")
        return final

    # ════════════════════════════════════════════════════════
    #   TAB: LANÇAR
    # ════════════════════════════════════════════════════════
    def _build_tab_launch(self) -> None:
        f = self.tab_launch

        # Cabeçalho
        top = tk.Frame(f, bg=BG)
        top.pack(fill=tk.X, padx=16, pady=10)
        tk.Label(top, text="🚀  Lançar Servidor llama-server",
                 bg=BG, fg=BLUE, font=FB).pack(side=tk.LEFT)

        # ── Modo: Lançar vs Conectar ──────────────────────
        mode_row = tk.Frame(f, bg=BG)
        mode_row.pack(fill=tk.X, padx=16, pady=(0, 6))
        tk.Label(mode_row, text="Modo:", bg=BG, fg=FG2, font=FS).pack(side=tk.LEFT)
        tk.Radiobutton(mode_row, text="⚡  Lançar servidor local",
                       variable=self.launch_mode, value="launch",
                       bg=BG, fg=FG, selectcolor=BG3, font=FS,
                       activebackground=BG, activeforeground=FG,
                       command=self._on_mode_change).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(mode_row, text="🔗  Conectar a servidor existente",
                       variable=self.launch_mode, value="connect",
                       bg=BG, fg=FG, selectcolor=BG3, font=FS,
                       activebackground=BG, activeforeground=FG,
                       command=self._on_mode_change).pack(side=tk.LEFT, padx=8)

        # Frame de conexão (só visível em modo connect)
        self.conn_frame = tk.Frame(f, bg=BG2, padx=12, pady=8)
        tk.Label(self.conn_frame, text="Host:", bg=BG2, fg=FG2, font=FS).pack(side=tk.LEFT)
        tk.Entry(self.conn_frame, textvariable=self.conn_host, width=18,
                 bg=BG3, fg=FG, font=FM, relief="flat", insertbackground=FG
                 ).pack(side=tk.LEFT, padx=4)
        tk.Label(self.conn_frame, text="Porta:", bg=BG2, fg=FG2, font=FS).pack(side=tk.LEFT, padx=(12,4))
        tk.Entry(self.conn_frame, textvariable=self.conn_port, width=8,
                 bg=BG3, fg=FG, font=FM, relief="flat", insertbackground=FG
                 ).pack(side=tk.LEFT, padx=4)
        self.conn_health_var = tk.StringVar(value="")
        tk.Label(self.conn_frame, textvariable=self.conn_health_var,
                 bg=BG2, fg=FG2, font=FS).pack(side=tk.LEFT, padx=12)
        ttk.Button(self.conn_frame, text="✔  Testar",
                   style="Blue.TButton",
                   command=self._test_connection).pack(side=tk.LEFT, padx=4)

        # Preview do comando
        self.cmd_frame = tk.Frame(f, bg=BG2, padx=12, pady=10)
        self.cmd_frame.pack(fill=tk.X, padx=16, pady=4)
        tk.Label(self.cmd_frame, text="Comando que será executado:", bg=BG2, fg=FG2, font=FS).pack(anchor="w")
        self.cmd_txt = tk.Text(
            self.cmd_frame, height=5, bg=BG4, fg=CYAN, font=FS,
            relief="flat", wrap=tk.WORD, state=tk.DISABLED,
            highlightthickness=0, padx=8, pady=6
        )
        self.cmd_txt.pack(fill=tk.X, pady=4)

        # URLs do servidor
        url_row = tk.Frame(f, bg=BG)
        url_row.pack(fill=tk.X, padx=16, pady=2)
        tk.Label(url_row, text="UI local:", bg=BG, fg=FG2, font=FS).pack(side=tk.LEFT)
        self.url_var = tk.StringVar(value="http://127.0.0.1:8080")
        url_lbl = tk.Label(url_row, textvariable=self.url_var,
                           bg=BG, fg=BLUE, font=FM, cursor="hand2")
        url_lbl.pack(side=tk.LEFT, padx=8)
        url_lbl.bind("<Button-1>", lambda e: webbrowser.open(self.url_var.get()))

        tk.Label(url_row, text="OpenAI compat:", bg=BG, fg=FG2, font=FS).pack(side=tk.LEFT, padx=(20,0))
        self.oai_var = tk.StringVar(value="http://127.0.0.1:8080/v1")
        tk.Label(url_row, textvariable=self.oai_var,
                 bg=BG, fg=TEAL, font=FM).pack(side=tk.LEFT, padx=8)

        # ── MCP Server ─────────────────────────────────────────
        mcp_row = tk.Frame(f, bg=BG)
        mcp_row.pack(fill=tk.X, padx=16, pady=(4, 0))
        cb = tk.Checkbutton(mcp_row, text="MCP LOCAL INTEGRADO",
                            variable=self.mcp_enabled,
                            command=self._on_mcp_toggle,
                            bg=BG, fg=FG, selectcolor=BG3, font=FS,
                            activebackground=BG, activeforeground=FG)
        cb.pack(side=tk.LEFT)
        tk.Label(mcp_row, text="stdio // projeto atual // sem rede",
                 bg=BG, fg=MAGENTA, font=FS).pack(side=tk.LEFT, padx=14)
        self.mcp_url_var = tk.StringVar(value="llama-server gerencia o ciclo de vida")

        # ── Autonomous mode toggle ─────────────────────────────
        self.autonomous_var = tk.BooleanVar(value=self._load_snn_enabled())
        tk.Checkbutton(mcp_row, text="SNN LOCAL",
                       variable=self.autonomous_var,
                       command=self._toggle_autonomous,
                       bg=BG, fg=GREEN, selectcolor=BG3, font=FS,
                       activebackground=BG, activeforeground=GREEN
                       ).pack(side=tk.LEFT, padx=(20, 0))

        # Botões
        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill=tk.X, padx=16, pady=8)

        self.btn_start = ttk.Button(btn_row, text="⚡  Iniciar Servidor",
                                    style="Green.TButton", command=self._launch)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(btn_row, text="⬛  Parar Servidor",
                                   style="Red.TButton", command=self._stop,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_row, text="↻  Atualizar Preview",
                   style="Blue.TButton",
                   command=self._update_cmd_preview).pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_row, text="🗑  Limpar Log",
                   style="Gray.TButton",
                   command=self._clear_log).pack(side=tk.RIGHT, padx=4)

        # Status do servidor
        self.srv_status_var = tk.StringVar(value="⬛  Parado")
        self.srv_status_lbl = tk.Label(f, textvariable=self.srv_status_var,
                                       bg=BG, fg=RED, font=FB)
        self.srv_status_lbl.pack(anchor="w", padx=16, pady=(0, 2))

        # Status do MCP
        self.mcp_status_var = tk.StringVar(value="⬛  MCP: inativo")
        self.mcp_status_lbl = tk.Label(f, textvariable=self.mcp_status_var,
                                       bg=BG, fg=FG2, font=FS)
        self.mcp_status_lbl.pack(anchor="w", padx=16, pady=(0, 4))

        # Configuração efetiva confirmada pelo processo, não apenas o valor
        # solicitado na interface.
        effective = tk.Frame(f, bg=BG2, padx=12, pady=8,
                             highlightbackground=BORDER, highlightthickness=1)
        effective.pack(fill=tk.X, padx=16, pady=(2, 5))
        tk.Label(effective, text="CONFIGURAÇÃO EFETIVA /PROPS", bg=BG2,
                 fg=CYAN, font=FS).pack(side=tk.LEFT)
        self.effective_ctx_var = tk.StringVar(value="contexto: aguardando")
        self.effective_slots_var = tk.StringVar(value="slots: —")
        self.effective_model_var = tk.StringVar(value="modelo: —")
        for variable in (
            self.effective_ctx_var, self.effective_slots_var,
            self.effective_model_var,
        ):
            tk.Label(effective, textvariable=variable, bg=BG2, fg=FG,
                     font=FS).pack(side=tk.LEFT, padx=(18, 0))

        # Log
        log_f = tk.Frame(f, bg=BG2, padx=4, pady=4)
        log_f.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)
        tk.Label(log_f, text="Output do servidor:", bg=BG2, fg=FG2, font=FS).pack(anchor="w")

        self.log_txt = scrolledtext.ScrolledText(
            log_f, bg=BG4, fg=TEAL, font=FS, relief="flat",
            wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0
        )
        self.log_txt.pack(fill=tk.BOTH, expand=True)
        self.log_txt.tag_configure("err",   foreground=RED)
        self.log_txt.tag_configure("warn",  foreground=YELLOW)
        self.log_txt.tag_configure("ok",    foreground=TEAL)
        self.log_txt.tag_configure("info",  foreground=CYAN)
        self.log_txt.tag_configure("dim",   foreground=FG2)
        self.log_txt.tag_configure("mag",   foreground=MAGENTA)

    def _on_mode_change(self) -> None:
        self._connect_request += 1
        self._test_request += 1
        if self.launch_mode.get() == "connect":
            self.conn_frame.pack(fill=tk.X, padx=16, pady=4, before=self.cmd_frame)
            self.btn_start.configure(text="🔗  Conectar")
        else:
            if self.connected:
                self.connected = False
                self._stop_mcp()
                self.srv_status_var.set("⬛  Parado")
                self.srv_status_lbl.configure(fg=RED)
                self.conn_health_var.set("")
            running = self.proc and self.proc.poll() is None
            self.btn_start.configure(state=tk.DISABLED if running else tk.NORMAL)
            self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)
            self.conn_frame.pack_forget()
            self.btn_start.configure(text="⚡  Iniciar Servidor")
        self._update_cmd_preview()

    def _test_connection(self) -> None:
        h = self.conn_host.get().strip() or "127.0.0.1"
        try:
            p = int(self.conn_port.get())
            if not 1 <= p <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("Porta inválida", "Informe uma porta entre 1 e 65535.")
            return
        base = f"http://{h}:{p}"
        self.conn_health_var.set("Testando...")
        self._test_request += 1
        request_id = self._test_request
        threading.Thread(
            target=self._test_connection_thread, args=(request_id, base), daemon=True
        ).start()

    def _test_connection_thread(self, request_id: int, base: str) -> None:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=4) as r:
                status = r.status
            self._post_ui(self._test_connection_done, request_id, f"✔  online (HTTP {status})")
        except Exception as exc:
            self._post_ui(self._test_connection_done, request_id, f"⚠  offline ({exc})")

    def _test_connection_done(self, request_id: int, status: str) -> None:
        if request_id == self._test_request:
            self.conn_health_var.set(status)

    def _connect(self) -> None:
        if self.connected:
            messagebox.showwarning("Aviso", "Já conectado a um servidor.")
            return
        h = self.conn_host.get().strip() or "127.0.0.1"
        try:
            p = int(self.conn_port.get())
            if not 1 <= p <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("Porta inválida", "Informe uma porta entre 1 e 65535.")
            return
        base = f"http://{h}:{p}"
        self._update_cmd_preview()
        self._log(f"\n{'═'*64}\n", "dim")
        self._log(f"🔗  Conectando a servidor existente: {base}\n", "ok")
        self._connect_request += 1
        request_id = self._connect_request
        self.conn_health_var.set("Conectando...")
        threading.Thread(
            target=self._connect_thread, args=(request_id, base, h, p), daemon=True
        ).start()

    def _connect_thread(self, request_id: int, base: str, host: str, port: int) -> None:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=4) as r:
                status = r.status
            self._post_ui(self._connect_done, request_id, base, host, port, status, "")
        except Exception as e:
            self._post_ui(self._connect_done, request_id, base, host, port, None, str(e))

    def _connect_done(self, request_id: int, base: str, h: str, p: int,
                      http_status, error: str) -> None:
        if request_id != self._connect_request or self.launch_mode.get() != "connect":
            return
        if error:
            self.conn_health_var.set("⚠  sem resposta")
            self._log(f"⚠  Servidor não respondeu em /health: {error}\n", "warn")
            if not messagebox.askyesno("Servidor offline?",
                                        f"Não foi possível alcançar {base}/health.\n"
                                        "Conectar mesmo assim?"):
                return
        else:
            self.conn_health_var.set(f"✔  online (HTTP {http_status})")
            self._log(f"✔  Servidor respondeu (HTTP {http_status})\n", "ok")
        self.connected = True
        self.url_var.set(base)
        self.oai_var.set(f"{base}/v1")
        try:
            mp = int(self.mcp_port.get())
        except Exception:
            mp = MCP_PORT
        self.mcp_url_var.set("stdio // llama-server")
        self.srv_status_var.set(f"🔗  Conectado: {base}")
        self.srv_status_lbl.configure(fg=GREEN)
        self.header_state_var.set("◇ CONNECTED")
        self.status_var.set(f"🔗  Conectado a {base}")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        try:
            webbrowser.open(base)
        except Exception:
            pass
        if self.mcp_enabled.get() and self.mcp_auto.get():
            self.root.after(200, self._start_mcp)

    def _update_cmd_preview(self) -> None:
        self.cmd_txt.configure(state=tk.NORMAL)
        self.cmd_txt.delete("1.0", tk.END)
        if self.launch_mode.get() == "connect":
            h = self.conn_host.get().strip() or "127.0.0.1"
            try:
                p = int(self.conn_port.get())
            except Exception:
                p = 8080
            self.url_var.set(f"http://{h}:{p}")
            self.oai_var.set(f"http://{h}:{p}/v1")
            try:
                mp = int(self.mcp_port.get())
            except Exception:
                mp = MCP_PORT
            self.mcp_url_var.set("stdio // indisponível em modo conexão")
            self.cmd_txt.insert(tk.END,
                "🔗  MODO CONEXÃO — nenhum comando será executado localmente.\n"
                f"   Alvo: http://{h}:{p}\n"
                "   Clique em 'Conectar' para validar o /health e abrir a UI.")
            self.cmd_txt.configure(state=tk.DISABLED)
            return
        if not self.opt:
            self.cmd_txt.insert(tk.END, "← Selecione um modelo na aba Modelo primeiro.")
            self.cmd_txt.configure(state=tk.DISABLED)
            return
        try:
            final = self._get_final()
            cmd = self.opt.build_cmd(final)
        except ValueError as exc:
            self.cmd_txt.insert(tk.END, f"⚠ Parâmetro inválido: {exc}")
            self.cmd_txt.configure(state=tk.DISABLED)
            return
        self.cmd_txt.insert(tk.END, self._display_command(cmd))
        self.cmd_txt.configure(state=tk.DISABLED)
        h = final["host"]; p = final["port"]
        self.url_var.set(f"http://{h}:{p}")
        self.oai_var.set(f"http://{h}:{p}/v1")
        try:
            mp = int(self.mcp_port.get())
        except Exception:
            mp = MCP_PORT
        self.mcp_url_var.set("stdio // llama-server")

    def _clear_log(self) -> None:
        self.log_txt.configure(state=tk.NORMAL)
        self.log_txt.delete("1.0", tk.END)
        self.log_txt.configure(state=tk.DISABLED)

    @staticmethod
    def _display_command(cmd: list[str]) -> str:
        hidden_after = {"--api-key", "--mcp-servers-json", "--mcp-servers-config"}
        values = []
        hide_next = False
        for arg in cmd:
            if hide_next:
                values.append("<configuração local protegida>")
                hide_next = False
                continue
            values.append(str(arg))
            hide_next = str(arg) in hidden_after
        return shlex.join(values)

    def _log(self, text: str, tag: str = "") -> None:
        self.log_txt.configure(state=tk.NORMAL)
        self.log_txt.insert(tk.END, text, tag if tag else ())
        line_count = int(self.log_txt.index("end-1c").split(".")[0])
        if line_count > 5000:
            self.log_txt.delete("1.0", f"{line_count - 4500}.0")
        self.log_txt.see(tk.END)
        self.log_txt.configure(state=tk.DISABLED)

    @staticmethod
    def _terminate_process(proc) -> None:
        if not proc or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except OSError:
                return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    # ── MCP Server Lifecycle ──────────────────────────────────
    def _legacy_on_mcp_toggle(self) -> None:
        if self.mcp_enabled.get():
            if self.connected or (self.proc and self.proc.poll() is None):
                self._start_mcp()
        else:
            self._stop_mcp()

    def _legacy_start_mcp(self) -> bool:
        if self.mcp_proc and self.mcp_proc.poll() is None:
            self._log(f"[MCP] Já está em execução (PID {self.mcp_proc.pid})\n", "mag")
            return True
        if not os.path.isfile(MCP_ENTRY):
            self._log(f"[MCP] entry-point não encontrado: {MCP_ENTRY}\n", "err")
            return False
        if self.launch_mode.get() == "launch" and not (
            self.proc and self.proc.poll() is None
        ):
            self._log("[MCP] Servidor local não está em execução.\n", "warn")
            return False
        port = MCP_PORT
        try:
            port = int(self.mcp_port.get())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            self._log("[MCP] Porta inválida; use um valor entre 1 e 65535.\n", "err")
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        except OSError as exc:
            self._log(f"[MCP] Porta {port} indisponível: {exc}\n", "err")
            return False
        cmd = ["node", MCP_ENTRY]
        env = os.environ.copy()
        env["MCP_PORT"] = str(port)
        try:
            self.mcp_proc = subprocess.Popen(
                cmd, cwd=MCP_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True
            )
        except Exception as e:
            self._log(f"[MCP] ERRO ao iniciar: {e}\n", "err")
            return False
        proc = self.mcp_proc
        self._log(f"[MCP] Iniciado (PID {proc.pid}) na porta {port}\n", "mag")
        self.mcp_status_var.set(f"🟣  MCP: iniciando (:{port})")
        self.mcp_status_lbl.configure(fg=MAGENTA)
        threading.Thread(target=self._read_mcp_output, args=(proc, port), daemon=True).start()
        self.root.after(1000, self._mcp_ready, proc, port)
        # Sync autonomous state after a brief delay for MCP to initialize
        self.root.after(2000, self._toggle_autonomous)
        return True

    def _legacy_stop_mcp(self) -> None:
        proc = self.mcp_proc
        self.mcp_proc = None
        if proc and proc.poll() is None:
            self._log("[MCP] Encerrando...\n", "mag")
            threading.Thread(target=self._terminate_process, args=(proc,), daemon=True).start()
        self.mcp_status_var.set("⬛  MCP: inativo")
        self.mcp_status_lbl.configure(fg=FG2)

    def _read_mcp_output(self, proc, port: int) -> None:
        rc = 0
        try:
            if proc and proc.stdout:
                for line in proc.stdout:
                    ll = line.lower()
                    tag = "dim"
                    if any(x in ll for x in ("error","failed","fatal","errno")):
                        tag = "err"
                    elif any(x in ll for x in ("warning","warn")):
                        tag = "warn"
                    elif "listening" in ll or "ready" in ll or "active" in ll:
                        tag = "ok"
                        self._post_ui(self._mcp_ready, proc, port)
                    self._post_ui(self._log, f"[MCP] {line}", tag)
        except (OSError, ValueError) as exc:
            self._post_ui(self._log, f"[MCP] erro ao ler saída: {exc}\n", "err")
        if proc:
            try:
                rc = proc.wait(timeout=30)
            except Exception:
                rc = -1
        self._post_ui(self._mcp_stopped, proc, rc)

    def _mcp_ready(self, proc, port: int) -> None:
        if proc is self.mcp_proc and proc.poll() is None:
            self.mcp_status_var.set(f"🟣  MCP: rodando (:{port})")

    def _mcp_stopped(self, proc, rc: int) -> None:
        if proc is not self.mcp_proc:
            return
        self.mcp_status_var.set(f"⬛  MCP: parado (exit: {rc})")
        self.mcp_status_lbl.configure(fg=RED if rc else FG2)
        self.mcp_proc = None

    def _legacy_toggle_autonomous(self) -> None:
        enabled = self.autonomous_var.get()
        try:
            p = int(self.mcp_port.get())
        except Exception:
            p = MCP_PORT
        threading.Thread(
            target=self._toggle_autonomous_thread, args=(enabled, p), daemon=True
        ).start()

    def _toggle_autonomous_thread(self, enabled: bool, port: int) -> None:
        url = f"http://127.0.0.1:{port}/agent/autonomous"
        try:
            import urllib.request
            req = urllib.request.Request(
                url, data=json.dumps({"enabled": enabled}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
            self._post_ui(self._log, f"[AUTONOMOUS] Modo {'ATIVADO' if enabled else 'DESATIVADO'}\n", "mag")
        except Exception:
            self._post_ui(self._log, "[AUTONOMOUS] MCP indisponível, modo será aplicado na próxima inicialização\n", "dim")

    # As definições abaixo substituem o lifecycle HTTP legado acima. O MCP
    # moderno é filho stdio do llama-server e pertence exclusivamente a este
    # projeto; nenhum processo ou diretório externo é consultado.
    def _on_mcp_toggle(self) -> None:
        self._save_settings()
        self._update_cmd_preview()
        if self.proc and self.proc.poll() is None:
            self._log(
                "[MCP] alteração será aplicada ao reiniciar o llama-server.\n",
                "warn",
            )
        self._start_mcp() if self.mcp_enabled.get() else self._stop_mcp()

    def _start_mcp(self) -> bool:
        if not LOCAL_MCP_ENTRY.is_file():
            self._log(
                f"[MCP] entry-point local não encontrado: {LOCAL_MCP_ENTRY}\n", "err"
            )
            return False
        running = self.proc and self.proc.poll() is None
        detail = "integrado ao llama-server" if running else "preparado para o próximo início"
        self.mcp_status_var.set(f"◇  MCP LOCAL: {detail}")
        self.mcp_status_lbl.configure(fg=MAGENTA)
        return True

    def _stop_mcp(self) -> None:
        self.mcp_proc = None
        self.mcp_status_var.set("◇  MCP LOCAL: desativado")
        self.mcp_status_lbl.configure(fg=FG2)

    def _toggle_autonomous(self) -> None:
        enabled = bool(self.autonomous_var.get())
        temporary = LOCAL_SNN_ENABLED.with_name(LOCAL_SNN_ENABLED.name + ".tmp")
        try:
            LOCAL_SNN_ENABLED.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps({"enabled": enabled, "updated_at": _time.time()}),
                encoding="utf-8",
            )
            os.replace(temporary, LOCAL_SNN_ENABLED)
            self._log(
                f"[SNN] núcleo local {'ATIVADO' if enabled else 'DESATIVADO'}\n",
                "mag",
            )
        except OSError as exc:
            try:
                temporary.unlink()
            except OSError:
                pass
            self._log(f"[SNN] falha ao persistir estado: {exc}\n", "err")

    def _launch(self) -> None:
        if self.launch_mode.get() == "connect":
            self._connect()
            return
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("Aviso", "Servidor já está em execução.")
            return
        if not self.opt:
            messagebox.showwarning("Aviso", "Selecione um modelo primeiro.")
            return
        if self._recalc_after is not None or self._recalc_running:
            messagebox.showinfo(
                "Cálculo em andamento",
                "Aguarde o recálculo de VRAM/contexto terminar antes de iniciar.",
            )
            return
        if not os.path.isfile(LLAMA_SERVER):
            messagebox.showerror("Erro",
                                 f"Executável não encontrado:\n{LLAMA_SERVER}\n\n"
                                 "Ajuste a variável LLAMA_SERVER no topo do script.")
            return

        self._update_cmd_preview()
        try:
            final = self._get_final()
            cmd = self.opt.build_cmd(final)
        except ValueError as exc:
            messagebox.showerror("Parâmetro inválido", str(exc))
            return
        self._log(f"\n{'═'*64}\n", "dim")
        self._log(f"⚡  Iniciando: {os.path.basename(self.model_path)}\n", "ok")
        self._log(f"   {self._display_command(cmd)}\n\n", "dim")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True
            )
        except Exception as e:
            self._log(f"ERRO ao lançar o servidor: {e}\n", "err")
            return

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.srv_status_var.set("◇  Carregando modelo; aguardando /health...")
        self.srv_status_lbl.configure(fg=YELLOW)
        self.header_state_var.set("◇ LOADING")
        self.status_var.set(
            f"Carregando servidor local em http://{final['host']}:{final['port']}...")
        self.effective_ctx_var.set(f"contexto solicitado: {final['ctx']}")
        self.effective_slots_var.set("slots: aguardando")
        self.effective_model_var.set(f"modelo: {Path(self.model_path).name}")

        proc = self.proc
        threading.Thread(target=self._read_output, args=(proc,), daemon=True).start()
        self._runtime_request += 1
        runtime_request = self._runtime_request
        threading.Thread(
            target=self._wait_for_server_ready,
            args=(runtime_request, proc, final), daemon=True,
        ).start()

    @staticmethod
    def _runtime_base(host: str, port: int) -> str:
        host = str(host or "127.0.0.1")
        if host in {"0.0.0.0", "localhost"}:
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{int(port)}"

    @staticmethod
    def _fetch_local_json(url: str, timeout: float = 2.0) -> dict:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("resposta JSON inválida")
        return value

    def _wait_for_server_ready(self, request_id: int, proc, final: dict) -> None:
        base = self._runtime_base(final["host"], final["port"])
        deadline = _time.monotonic() + max(60, min(int(final["timeout"]), 900))
        last_error = "servidor ainda não respondeu"
        while _time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            try:
                self._fetch_local_json(base + "/health")
                props = self._fetch_local_json(base + "/props")
                generation = props.get("default_generation_settings", {})
                context = int(generation.get("n_ctx") or final["ctx"])
                slots = int(props.get("total_slots") or final["parallel"] or 1)
                model = str(props.get("model_path") or final["model_path"])
                self._post_ui(
                    self._server_ready, request_id, proc, base, context, slots, model
                )
                return
            except Exception as exc:
                last_error = str(exc)
                _time.sleep(0.25)
        self._post_ui(self._server_ready_failed, request_id, proc, last_error)

    def _server_ready(self, request_id: int, proc, base: str, context: int,
                      slots: int, model: str) -> None:
        if request_id != self._runtime_request or proc is not self.proc:
            return
        self.srv_status_var.set(f"◇  RUNNING  //  {base}")
        self.srv_status_lbl.configure(fg=GREEN)
        self.header_state_var.set("◇ RUNNING")
        self.status_var.set(
            f"✔  Runtime confirmado: contexto {context}, {slots} slot(s)."
        )
        self.effective_ctx_var.set(f"contexto: {context}")
        self.effective_slots_var.set(f"slots: {slots}")
        self.effective_model_var.set(f"modelo: {Path(model).name}")
        self._log(
            f"[runtime confirmado por /props: contexto={context}, slots={slots}]\n",
            "ok",
        )
        if self.mcp_enabled.get() and self.mcp_auto.get():
            self.root.after(200, self._start_mcp)

    def _server_ready_failed(self, request_id: int, proc, error: str) -> None:
        if request_id != self._runtime_request or proc is not self.proc:
            return
        self.header_state_var.set("◇ ERROR")
        self.srv_status_var.set("⚠  Servidor não ficou pronto")
        self.srv_status_lbl.configure(fg=RED)
        self._log(f"[falha de readiness: {error}]\n", "err")
        threading.Thread(target=self._terminate_process, args=(proc,), daemon=True).start()

    def _read_output(self, proc):
        try:
            if proc.stdout:
                for line in proc.stdout:
                    ll = line.lower()
                    if any(x in ll for x in ("error","failed","fatal","errno")):
                        tag = "err"
                    elif any(x in ll for x in ("warning","warn")):
                        tag = "warn"
                    elif any(x in ll for x in ("server listening","model loaded","llama server")):
                        tag = "ok"
                    elif any(x in ll for x in ("ggml","llama_","cuda","metal")):
                        tag = "dim"
                    elif any(x in ll for x in ("slot","request","prompt","generation")):
                        tag = "info"
                    else:
                        tag = ""
                    self._post_ui(self._log, line, tag)
        except (OSError, ValueError) as exc:
            self._post_ui(self._log, f"Erro ao ler saída do servidor: {exc}\n", "err")
        rc = 0
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rc = -1
        self._post_ui(self._srv_stopped, proc, rc)

    def _stop(self) -> None:
        self._stop_mcp()
        if self.launch_mode.get() == "connect" and self.connected:
            self.connected = False
            self.srv_status_var.set("⬛  Desconectado")
            self.srv_status_lbl.configure(fg=FG2)
            self.header_state_var.set("◇ IDLE")
            self.status_var.set("Desconectado do servidor.")
            self.btn_start.configure(state=tk.NORMAL)
            self.btn_stop.configure(state=tk.DISABLED)
            self.conn_health_var.set("")
            self._log("\n🔌  Desconectado do servidor remoto.\n", "warn")
            return
        if self.proc and self.proc.poll() is None:
            proc = self.proc
            self.btn_stop.configure(state=tk.DISABLED)
            self.srv_status_var.set("⬛  Encerrando servidor...")
            threading.Thread(target=self._terminate_process, args=(proc,), daemon=True).start()
            self._log("\n⬛  Servidor encerrado pelo usuário.\n", "warn")

    def _srv_stopped(self, proc, rc: int) -> None:
        if proc is not self.proc:
            return
        self.proc = None
        self._stop_mcp()
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.srv_status_var.set(f"⬛  Parado  (exit code: {rc})")
        self.srv_status_lbl.configure(fg=RED if rc else FG2)
        self.header_state_var.set("◇ IDLE")
        self.effective_ctx_var.set("contexto: aguardando")
        self.effective_slots_var.set("slots: —")
        self.status_var.set("Servidor parado.")
        if rc:
            self._log(f"\n⚠  Servidor encerrou com código {rc}\n", "err")


def main() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()

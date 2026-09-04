#!/usr/bin/env bash
set -euo pipefail

# Gerencia exclusivamente o swap NVMe do Crono Matrix. O caminho e derivado
# do usuario que abriu o launcher; nenhum alvo arbitrario e aceito.

action="${1:-status}"
size_gib="${2:-32}"
caller_uid="${PKEXEC_UID:-${SUDO_UID:-$(id -u)}}"
caller_home="$(getent passwd "$caller_uid" | cut -d: -f6)"

if [[ -z "$caller_home" || "$caller_home" == "/" ]]; then
    echo "erro=nao foi possivel determinar o diretorio pessoal do usuario" >&2
    exit 2
fi

swap_dir="$caller_home/.local/share/crono-matrix"
swap_file="$swap_dir/swapfile"
fstab_file="/etc/fstab"

# O swap que protege uma carga grande precisa ser escolhido antes da ZRAM.
# Caso contrario o kernel comprime dezenas de GiB dentro da propria RAM e so
# alcanca o NVMe depois de esgotar a ZRAM, exatamente quando a maquina ja esta
# sob thrashing severo. Derive a prioridade da tabela ativa em vez de assumir
# que toda distribuicao configura a ZRAM com o mesmo valor.
zram_priority="$(awk '$1 ~ /^\/dev\/zram/ {if (!seen || $5 > max) max=$5; seen=1} END {if (seen) print max}' /proc/swaps)"
zram_priority="${zram_priority:-100}"
if (( zram_priority >= 32767 )); then
    swap_priority=32767
else
    swap_priority=$((zram_priority + 10))
fi
fstab_line="$swap_file none swap defaults,pri=$swap_priority 0 0"

is_active() {
    swapon --noheadings --show=NAME 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -Fqx "$swap_file"
}

print_status() {
    local active="no" size_bytes="0" used_bytes="0" priority="-1"
    if is_active; then
        active="yes"
        read -r size_kib used_kib priority < <(
            awk -v target="$swap_file" '$1 == target {print $3, $4, $5}' /proc/swaps
        )
        size_bytes=$((size_kib * 1024))
        used_bytes=$((used_kib * 1024))
    elif [[ -f "$swap_file" ]]; then
        size_bytes="$(stat -c '%s' "$swap_file" 2>/dev/null || printf '0')"
    fi
    printf 'path=%s\nactive=%s\nsize_bytes=%s\nused_bytes=%s\npriority=%s\nzram_priority=%s\ndesired_priority=%s\n' \
        "$swap_file" "$active" "$size_bytes" "$used_bytes" "$priority" \
        "$zram_priority" "$swap_priority"
}

update_fstab() {
    local temporary
    if [[ ! -e "$fstab_file.crono-matrix.bak" ]]; then
        cp -a "$fstab_file" "$fstab_file.crono-matrix.bak"
    fi
    temporary="$(mktemp /etc/fstab.crono-matrix.XXXXXX)"
    awk -v target="$swap_file" '$1 != target' "$fstab_file" > "$temporary"
    printf '%s\n' "$fstab_line" >> "$temporary"
    chmod --reference="$fstab_file" "$temporary"
    chown --reference="$fstab_file" "$temporary"
    mv "$temporary" "$fstab_file"
}

if [[ "$action" == "status" ]]; then
    print_status
    exit 0
fi

if [[ "$EUID" -ne 0 ]]; then
    echo "erro=esta operacao requer autenticacao administrativa" >&2
    exit 3
fi

case "$action" in
    create)
        if [[ ! "$size_gib" =~ ^[0-9]+$ ]] || (( size_gib < 8 || size_gib > 64 )); then
            echo "erro=tamanho deve estar entre 8 e 64 GiB" >&2
            exit 4
        fi
        if (( zram_priority >= 32767 )); then
            echo "erro=a ZRAM ja usa a prioridade maxima 32767; reduza a prioridade da ZRAM antes de ativar o swap Crono" >&2
            exit 8
        fi

        install -d -m 0755 "$swap_dir"
        filesystem="$(findmnt -n -o FSTYPE -T "$swap_dir")"
        source_device="$(findmnt -n -o SOURCE -T "$swap_dir")"
        if [[ "$filesystem" != "btrfs" ]]; then
            echo "erro=o gerenciador exige Btrfs; detectado $filesystem" >&2
            exit 5
        fi
        if [[ "$source_device" != *nvme* ]]; then
            echo "erro=o alvo nao esta em NVMe: $source_device" >&2
            exit 6
        fi

        available_bytes="$(df -B1 --output=avail "$swap_dir" | awk 'NR == 2 {print $1}')"
        requested_bytes=$((size_gib * 1024 * 1024 * 1024))
        reserve_bytes=$((32 * 1024 * 1024 * 1024))
        reclaimable_bytes=0
        if [[ -f "$swap_file" ]]; then
            reclaimable_bytes="$(stat -c '%s' "$swap_file")"
        fi
        if (( available_bytes + reclaimable_bytes - requested_bytes < reserve_bytes )); then
            echo "erro=espaco insuficiente; preserve pelo menos 32 GiB livres apos criar o swap" >&2
            exit 7
        fi

        if is_active; then
            current_kib="$(awk -v target="$swap_file" '$1 == target {print $3}' /proc/swaps)"
            current_priority="$(awk -v target="$swap_file" '$1 == target {print $5}' /proc/swaps)"
            current_bytes=$((current_kib * 1024))
            # Btrfs reserves one page for the swap header, so a nominal 16 GiB
            # file is reported by /proc/swaps as 16 GiB minus 4 KiB.
            difference=$((current_bytes - requested_bytes))
            (( difference < 0 )) && difference=$((-difference))
            if (( difference <= 1048576 )); then
                if (( current_priority != swap_priority )); then
                    swapoff "$swap_file"
                    if ! swapon -p "$swap_priority" "$swap_file"; then
                        # Nao deixe um swap antes ativo desabilitado se a nova
                        # ativacao falhar por qualquer motivo.
                        swapon -p "$current_priority" "$swap_file" || true
                        echo "erro=nao foi possivel reaplicar o swap com prioridade $swap_priority" >&2
                        exit 9
                    fi
                fi
                update_fstab
                print_status
                exit 0
            fi
            swapoff "$swap_file"
        fi

        # Apenas o caminho fixo acima pode ser substituido.
        if [[ -e "$swap_file" ]]; then
            rm -f -- "$swap_file"
        fi
        btrfs filesystem mkswapfile --size "${size_gib}G" "$swap_file"
        swapon -p "$swap_priority" "$swap_file"

        update_fstab
        print_status
        ;;
    remove)
        if is_active; then
            swapoff "$swap_file"
        fi
        if [[ -f "$fstab_file" ]]; then
            temporary="$(mktemp /etc/fstab.crono-matrix.XXXXXX)"
            awk -v target="$swap_file" '$1 != target' "$fstab_file" > "$temporary"
            chmod --reference="$fstab_file" "$temporary"
            chown --reference="$fstab_file" "$temporary"
            mv "$temporary" "$fstab_file"
        fi
        if [[ -e "$swap_file" ]]; then
            rm -f -- "$swap_file"
        fi
        print_status
        ;;
    *)
        echo "uso: $0 {status|create [8-64]|remove}" >&2
        exit 2
        ;;
esac

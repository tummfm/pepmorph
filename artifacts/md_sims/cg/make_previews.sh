#!/usr/bin/env bash
set -u -o pipefail

export LC_ALL=C
export LANG=C

GMX="${GMX:-gmx}"

find_topology() {
    local xtc="$1"
    local dir stem
    dir="$(dirname "$xtc")"
    stem="$(basename "$xtc" .xtc)"

    for ext in tpr gro pdb; do
        if [[ -f "$dir/$stem.$ext" ]]; then
            echo "$dir/$stem.$ext"
            return 0
        fi
    done

    for ext in tpr gro pdb; do
        local candidate=""
        candidate="$(find "$dir" -maxdepth 1 -type f -name "*.$ext" | head -n 1 || true)"
        if [[ -n "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

process_xtc() {
    local xtc="$1"
    local dir out top info nframes dt t0 tmpdir
    dir="$(dirname "$xtc")"
    out="$dir/preview.xtc"

    [[ "$(basename "$xtc")" == "preview.xtc" ]] && return 0

    if ! top="$(find_topology "$xtc")"; then
        echo "[fail] $xtc -> no .tpr/.gro/.pdb found in same directory"
        return 1
    fi

    if ! info="$($GMX check -f "$xtc" 2>&1)"; then
        echo "[fail] $xtc -> gmx check failed"
        echo "$info"
        return 1
    fi

    nframes="$(awk '/^Coords[[:space:]]+[0-9]+/ { print $2; exit }' <<< "$info")"
    dt="$(awk '/^Coords[[:space:]]+[0-9]+/ { print $3; exit }' <<< "$info")"
    t0="$(awk '
        /Reading frame[[:space:]]+0/ {
            for (i=1; i<=NF; i++) {
                if ($i == "time") { print $(i+1); exit }
            }
        }
    ' <<< "$info")"

    if [[ -z "${nframes:-}" || -z "${dt:-}" || -z "${t0:-}" ]]; then
        echo "[fail] $xtc -> could not parse gmx check output"
        echo "$info"
        return 1
    fi

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' RETURN

    echo "[info] $xtc"
    echo "       topology: $top"
    echo "       nframes : $nframes"
    echo "       dt(ps)  : $dt"
    echo "       t0(ps)  : $t0"

    rm -f "$out"

    if (( nframes == 1 )); then
        if ! printf "0\n" | "$GMX" trjconv -s "$top" -f "$xtc" -o "$out" -dump "$t0" >"$tmpdir/oneframe.log" 2>&1; then
            echo "[fail] $xtc -> trjconv failed"
            cat "$tmpdir/oneframe.log"
            return 1
        fi

        if [[ ! -s "$out" ]]; then
            echo "[fail] $xtc -> output file was not created"
            cat "$tmpdir/oneframe.log"
            return 1
        fi

        echo "[ok]   $xtc -> $out"
        ls -lh "$out"
        return 0
    fi

    local -a times parts
    mapfile -t times < <(
        awk -v n="$nframes" -v dt="$dt" -v t0="$t0" '
            BEGIN {
                tlast = t0 + (n - 1) * dt
                if (n < 5) {
                    for (i = 0; i < n; i++) {
                        printf "%.6f\n", t0 + i * dt
                    }
                } else {
                    for (i = 0; i < 5; i++) {
                        printf "%.6f\n", t0 + i * (tlast - t0) / 4.0
                    }
                }
            }
        '
    )

    local i=0
    local part t
    for t in "${times[@]}"; do
        part="$tmpdir/frame_${i}.xtc"

        if ! printf "0\n" | "$GMX" trjconv -s "$top" -f "$xtc" -o "$part" -dump "$t" >"$tmpdir/trjconv_${i}.log" 2>&1; then
            echo "[fail] $xtc -> trjconv failed for time $t"
            cat "$tmpdir/trjconv_${i}.log"
            return 1
        fi

        if [[ ! -s "$part" ]]; then
            echo "[fail] $xtc -> frame file missing for time $t"
            cat "$tmpdir/trjconv_${i}.log"
            return 1
        fi

        parts+=("$part")
        ((i+=1))
    done

    if ! "$GMX" trjcat -f "${parts[@]}" -o "$out" -cat >"$tmpdir/trjcat.log" 2>&1; then
        echo "[fail] $xtc -> trjcat failed"
        cat "$tmpdir/trjcat.log"
        return 1
    fi

    if [[ ! -s "$out" ]]; then
        echo "[fail] $xtc -> preview.xtc was not created"
        cat "$tmpdir/trjcat.log"
        return 1
    fi

    echo "[ok]   $xtc -> $out"
    ls -lh "$out"
    return 0
}

export -f find_topology
export -f process_xtc
export GMX

find . -type f -name "*.xtc" ! -name "preview.xtc" -print0 |
while IFS= read -r -d '' xtc; do
    process_xtc "$xtc"
done
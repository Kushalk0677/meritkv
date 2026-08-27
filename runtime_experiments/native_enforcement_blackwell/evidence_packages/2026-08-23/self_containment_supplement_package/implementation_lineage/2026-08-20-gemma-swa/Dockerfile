FROM shadowkv-sglang-native-admission:2026-06-22-counters

COPY apply_swa_native_admission.py /tmp/apply_swa_native_admission.py
RUN python3 /tmp/apply_swa_native_admission.py \
    && python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/sglang/srt/mem_cache/swa_radix_cache.py \
    && rm /tmp/apply_swa_native_admission.py

LABEL org.opencontainers.image.title="MeritKV native admission with SWARadixCache counters" \
      meritkv.base_image_id="sha256:5734af3b7592b0f4d082101330e8190f55c31ca90911302023c839479aa05caf" \
      meritkv.patch="swa-radix-native-admission-v1" \
      meritkv.created="2026-08-20"

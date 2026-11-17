#!/bin/bash
# v8 teacher labels on one VM: P processes over the student windows' S shards, then the stage 4 and news sets;
# every shard uploads when done and is skipped if the hub has it, so a second VM picks up where the first stopped.
#   bash v6/train/teacher_labels.sh RUN P S MEM_FRACTION
RUN=$1; P=$2; S=$3; F=${4:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
one() {   # set shard shards
  $PY -m v6.train.teacher_labels dump --run $RUN --set $1 --shard $2/$3 --upload --skip-done --mem-fraction $F \
    > /content/tl-$1-$2.log 2>&1 || echo "tl $1 $2/$3 failed: $(tail -n 1 /content/tl-$1-$2.log)"
  grep -E "^  [0-9]|skipped|wrote|uploaded|Error" /content/tl-$1-$2.log | tail -n 2
}
worker() {   # first step
  for k in $(seq $1 $2 $((S - 1))); do one student $k $S; done
  for k in $(seq $1 $2 7); do one s4 $k 8; done
  for k in $(seq $1 $2 1); do one news $k 2; done
}
for t in 1 2; do       # a second pass reruns what failed (resuming from its saved records)
  for j in $(seq 0 $((P - 1))); do worker $j $P & done; wait
done
echo "teacher labels done: $RUN"

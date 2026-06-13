#!/usr/bin/env bash
# build_group.sh — 群(group)別 Stage2 細分類器を作る「段階型ドライバ」。
#
# 設計思想（docs/group_classifier_playbook.md と1対1）:
#   - 各段(stage)は独立に再実行可能。判断点ではドライバは「止まって情報を出す」だけで決めない
#     （薄種のworldwide補填 / grade緩和 / 複合クラス化 / OOD動作点 / 昇格可否 は人間が判断）。
#   - 群固有値は config-<group>.yaml から導出。新群は config を作って各段を順に回すだけ。
#
# 使い方:
#   scripts/build_group.sh <stage> --config config-<group>.yaml [opts]
#   stage: data | prep | embed | train | eval | ood | confusion | register
#
#   共通: --config <f>(必須)  --tag <T>(モデル名識別子, 例 AB)  --splits-dir <d>(既定=config)
#         --dry-run(発行コマンドを表示のみ)  --seeds "42 1 2"  --num-workers <N>
#   train: --arm lean|kd(既定 lean)
#   eval : --eval-splits-dir <d>(共通test, 既定=--splits-dir)
#   prep : --grade-ablate "<種>"(任意, §3 grade アブレーション split を生成)
#   confusion: --topn <N>(perch native confusion 上位, 既定 30)
#
# 例（カラスを Lean→KD で）:
#   scripts/build_group.sh data  --config config-crow.yaml
#   scripts/build_group.sh prep  --config config-crow.yaml
#   scripts/build_group.sh train --config config-crow.yaml --tag AB --arm lean --seeds "42 1 2"
#   scripts/build_group.sh embed --config config-crow.yaml --tag AB
#   scripts/build_group.sh train --config config-crow.yaml --tag AB --arm kd  --seeds "42 1 2"
#   scripts/build_group.sh eval  --config config-crow.yaml --tag AB --eval-splits-dir data/splits-crow-ABC

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # repo ルートへ

# --- インタプリタ（本体venv / perch専用venv）。uv 管理の .venv を直叩き（perch は別依存で隔離） ---
PY=${PY:-.venv/bin/python}
PERCHPY=${PERCHPY:-tools/perch_embed/.venv/bin/python}

# --- 引数パース ---
STAGE="${1:-}"; shift || true
CONFIG=""; TAG=""; SPLITS=""; EVAL_SPLITS=""; ARM="lean"
SEEDS="42 1 2"; NW=8; TOPN=30; GRADE_ABLATE=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config)          CONFIG="$2"; shift 2;;
    --tag)             TAG="$2"; shift 2;;
    --splits-dir)      SPLITS="$2"; shift 2;;
    --eval-splits-dir) EVAL_SPLITS="$2"; shift 2;;
    --arm)             ARM="$2"; shift 2;;
    --seeds)           SEEDS="$2"; shift 2;;
    --num-workers)     NW="$2"; shift 2;;
    --topn)            TOPN="$2"; shift 2;;
    --grade-ablate)    GRADE_ABLATE="$2"; shift 2;;
    --dry-run)         DRY=1; shift;;
    *) echo "不明な引数: $1" >&2; exit 2;;
  esac
done

die(){ echo "ERROR: $*" >&2; exit 1; }
log(){ echo "[build_group] $* $(date '+%m-%d %H:%M:%S')"; }
hr(){  echo "------------------------------------------------------------"; }
# run: DRY なら表示のみ、実行時はエコーしてから実行
run(){ echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

[ -n "$STAGE" ]  || die "stage 未指定（data|prep|embed|train|eval|ood|confusion|register）"
[ -n "$CONFIG" ] || die "--config 必須"
[ -f "$CONFIG" ] || die "config が無い: $CONFIG"

# --- 導出: GROUP（config-<group>.yaml → <group>, config.yaml → duck） ---
base="$(basename "$CONFIG" .yaml)"
case "$base" in
  config-*) GROUP="${base#config-}";;
  config)   GROUP="duck";;
  *)        GROUP="$base";;
esac
# SPLITS 既定 = config の preprocessing.splits_dir（簡易抽出, yamlライブラリ不要）
if [ -z "$SPLITS" ]; then
  SPLITS="$(grep -E '^[[:space:]]*splits_dir:' "$CONFIG" | head -1 | sed -E 's/.*:[[:space:]]*["'\'']?([^"'\'' ]+).*/\1/')"
fi
[ -n "$SPLITS" ] || die "splits_dir を config から取得できず。--splits-dir で指定して"
[ -z "$EVAL_SPLITS" ] && EVAL_SPLITS="$SPLITS"

# モデル名規約（既存 crow 成果物と互換: lean seed は arm トークン無し, kd は -kd）
TAGPART=""; [ -n "$TAG" ] && TAGPART="-$TAG"
model_seed(){ local arm="$1" s="$2" tok=""; [ "$arm" = kd ] && tok="-kd"; echo "models/ast-${GROUP}${TAGPART}${tok}-s${s}"; }
soup_dir(){  echo "models/ast-${GROUP}${TAGPART}-$1-soup"; }   # $1=arm
PERCH_EMB="data/embeddings/perch-${GROUP}"
TEACHER="data/embeddings/teacher_proba-${GROUP}"

log "stage=$STAGE group=$GROUP tag='${TAG:-}' splits=$SPLITS dry=$DRY"
hr

case "$STAGE" in

  # §2 データ収集（Japan→worldwide fallback）。収集後は止まって判断を促す。
  data)
    run $PY -m bird_fine.data.download --config "$CONFIG"
    hr; echo "判断ゲート(§2): 各種の収集件数を確認しなよ。"
    echo "  - 薄種(>0 で止まった種): $PY -m bird_fine.data.download --config $CONFIG --species \"<種>\" --worldwide-only --exclude-existing"
    echo "  - 地域不在種(Japan=0): worldwide 自動発動済。タクソンズレ(en検索の別種混入)を data/raw/<種>/ の学名サブディレクトリで確認"
    echo "  - grade緩和(A B C)は基本不要（分離容易群＋非枯渇では clean-test 改善せず）。まず worldwide A+B。"
    ;;

  # §3 前処理＋録音単位split（任意で grade アブレーション split）
  prep)
    run $PY -m bird_fine.data.preprocess --config "$CONFIG"
    run $PY -m bird_fine.data.split --config "$CONFIG"
    if [ -n "$GRADE_ABLATE" ]; then
      run $PY tools/make_grade_ablation_splits.py --config "$CONFIG" \
        --ablate "$GRADE_ABLATE" --out-prefix "$SPLITS"
      echo "→ A / A+B / A+B+C split を val/test 共通で生成（--tag で AB/ABC を使い分け）"
    fi
    ;;

  # §4(KD前処理) Perch埋め込み(train) + 教師proba(OOF 5fold)。クラス順は lean s42 から取る。
  embed)
    AST42="$(model_seed lean 42)"
    [ -d "$AST42" ] || [ "$DRY" = 1 ] || die "教師procのクラス順に lean s42 が要る($AST42)。先に train --arm lean --seeds 42"
    run $PERCHPY tools/perch_embed/extract_perch.py --splits-dir "$SPLITS" \
      --out-dir "$PERCH_EMB" --source raw --chunk-sec 3.0 --min-chunk-sec 1.0 --splits train
    run $PY tools/probe_sweep/export_teacher_proba.py \
      --emb-dir "$PERCH_EMB" --ast-dir "$AST42" \
      --out-dir "$TEACHER" --splits train --oof-folds 5
    ;;

  # §4 学習: --arm lean|kd ×複数seed → soup
  train)
    [ "$ARM" = lean ] || [ "$ARM" = kd ] || die "--arm は lean か kd"
    seedmodels=()
    for s in $SEEDS; do
      out="$(model_seed "$ARM" "$s")"
      if [ "$ARM" = kd ]; then
        run $PY -m bird_fine.training.train --config "$CONFIG" --splits-dir "$SPLITS" \
          --distill --kd-lambda 1.0 --kd-temp 2.0 \
          --teacher-dir "$TEACHER" --duck-order "$TEACHER/duck_order.csv" \
          --output-dir "$out" --seed "$s" --num-workers "$NW"
      else
        run $PY -m bird_fine.training.train --config "$CONFIG" --splits-dir "$SPLITS" \
          --output-dir "$out" --seed "$s" --num-workers "$NW"
      fi
      seedmodels+=("$out")
    done
    run $PY tools/soup_ast.py --out "$(soup_dir "$ARM")" "${seedmodels[@]}"
    ;;

  # §4 評価: 両armの soup を共通testでproba化 → 録音単位 bootstrap CI
  eval)
    for arm in lean kd; do
      sp="$(soup_dir "$arm")"
      [ -d "$sp" ] || [ "$DRY" = 1 ] || { echo "skip: $sp 未作成"; continue; }
      run $PY tools/ast_eval_proba.py --model-dir "$sp" \
        --config "$CONFIG" --splits-dir "$EVAL_SPLITS" --tag "${GROUP}_${arm}"
    done
    run $PY tools/probe_sweep/soup_ci.py --tags "${GROUP}_lean" "${GROUP}_kd" \
      --pairs "${GROUP}_kd:${GROUP}_lean"
    hr; echo "判断ゲート(§4): CIが0を跨ぐ差(≈±0.05)は『差なし』。既定はLean。KDはCIで有意な時だけ採用。"
    echo "  昇格は必ず候補が未学習の録音で(リーク厳禁)。"
    ;;

  # §6 OOD energy 閾値キャリブレ（動作点は人間が選ぶ）
  ood)
    sp="$(soup_dir kd)"; [ -d "$sp" ] || sp="$(soup_dir lean)"
    run $PY tools/ood_fp_audit.py --config "$CONFIG" --ast-model "$sp"
    hr; echo "判断ゲート(§6): 動作点『真${GROUP}最優先・保持≥0.90』で energy 閾値を選ぶ。"
    echo "  ⚠モデルを替えたら必ず再導出（energy分布シフト）。XC域の暫定→デプロイ後フィールド再キャリブレ必須。"
    ;;

  # §5 混同分析（複合クラス化は実測してから＝人間判断）
  confusion)
    run $PERCHPY tools/perch_native_confusion.py "$TOPN"
    hr; echo "判断ゲート(§5): Perch本体でも割れないペアだけ display_groups で複合化（再学習ゼロ）。"
    echo "  録音単位の混同行列は eval 出力に対して手動で:"
    echo "    $PY -m bird_fine.analysis.confusion_audio --eval-dir outputs/eval_<日時>"
    ;;

  # §7 登録（taxonomy 追記スニペットを出力するだけ。自動編集しない）
  register)
    sp="$(soup_dir kd)"; [ -d "$sp" ] || sp="$(soup_dir lean)"
    hr; echo "§7 species_taxonomy.yaml に以下を追記（BirdProject 側, 値は実測で確定後）:"
    cat <<EOF

  ${GROUP}:
    pipeline:
      stage2_model: "<bird-fine-classifier>/${sp}"
      energy_threshold: <ood で導出した値>
      energy_temperature: 1.0
      # display_groups: [["<複合表示>", ["<種A>", "<種B>"]]]   # confusion で必要なら

EOF
    echo "→ dispatch は群汎用。taxonomy 追記だけで stage2_refine.py が配線。predict.py --group ${GROUP} --json で検証。"
    ;;

  *) die "未知の stage: $STAGE";;
esac

hr; log "DONE stage=$STAGE"

# python scripts/linscan_validation.py --qindex 0 --qset guten
# python scripts/linscan_validation.py --qindex 1 --qset guten
# python scripts/linscan_validation.py --qindex 2 --qset guten
# python scripts/linscan_validation.py --qindex 3 --qset guten

# python scripts/linscan_validation.py --qindex 0 --qset wiki
# python scripts/linscan_validation.py --qindex 1 --qset wiki

# do for qwen 3 8B now
python scripts/linscan_validation.py --qindex 1 --qset guten --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 2 --qset guten --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 0 --qset guten --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 3 --qset guten --model Qwen/Qwen3-8B --mkey qwen38b

python scripts/linscan_validation.py --qindex 0 --qset wiki --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 1 --qset wiki --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 2 --qset wiki --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 3 --qset wiki --model Qwen/Qwen3-8B --mkey qwen38b
python scripts/linscan_validation.py --qindex 4 --qset wiki --model Qwen/Qwen3-8B --mkey qwen38b

# do for gemini-flash-lite now
python scripts/linscan_validation.py --qindex 0 --qset guten --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 1 --qset guten --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 2 --qset guten --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 3 --qset guten --model gemini-2.5-flash-lite --mkey geminiflashlite

python scripts/linscan_validation.py --qindex 0 --qset wiki --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 1 --qset wiki --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 2 --qset wiki --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 3 --qset wiki --model gemini-2.5-flash-lite --mkey geminiflashlite
python scripts/linscan_validation.py --qindex 4 --qset wiki --model gemini-2.5-flash-lite --mkey geminiflashlite

# python scripts/linscan_validation.py --qindex 2 --qset wiki
# python scripts/linscan_validation.py --qindex 3 --qset wiki
# python scripts/linscan_validation.py --qindex 4 --qset wiki

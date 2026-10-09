"""评估脚本入口。

用法：
    python eval/run_eval.py --queries eval/queries.jsonl --top-k 10
等价于：rag eval
"""

from wealth_rag.evaluation import main

if __name__ == "__main__":
    main()
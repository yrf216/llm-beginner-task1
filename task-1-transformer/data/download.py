"""下载 ChnSentiCorp 中文情感分类数据集到当前目录。

依赖：pip install datasets pyarrow huggingface_hub

优先直接下载 Hugging Face 仓库中的 Arrow 文件并转成 parquet。
如果国内访问 HF 不稳定：
  - 方案一（推荐）：设置环境变量 HF_ENDPOINT=https://hf-mirror.com
  - 方案二：用 ModelScope（见脚本末尾提示）
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ID = "seamew/ChnSentiCorp"
ARROW_FILES = {
    "train": "chn_senti_corp-train.arrow",
    "validation": "chn_senti_corp-validation.arrow",
    "test": "chn_senti_corp-test.arrow",
}


def main():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 如下载缓慢，可先 export HF_ENDPOINT=https://hf-mirror.com 再重试\n")

    try:
        from datasets import Dataset
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[错误] 缺少依赖：pip install datasets pyarrow huggingface_hub")

    print(f"正在下载 {REPO_ID} 的 Arrow 文件 ...")
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split, filename in ARROW_FILES.items():
        local_arrow = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        ds = Dataset.from_file(local_arrow)
        out = DATA_DIR / f"{split}.parquet"
        ds.to_parquet(str(out))
        print(f"  {split}: {len(ds)} 条 -> {out.name}")

    print(f"\n完成。数据保存在 {DATA_DIR}")
    print("\n--- ModelScope 备选下载 ---")
    print("如果你完全无法访问 HF：")
    print("  pip install modelscope")
    print("  python -c \"from modelscope.msdatasets import MsDataset; "
          "ds = MsDataset.load('ChnSentiCorp'); print(ds)\"")


if __name__ == "__main__":
    main()

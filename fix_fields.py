from pathlib import Path
import re

file_path = Path("fairseq-local/fairseq/data/indexed_dataset.py")  # 相对路径，或替换成绝对路径

if file_path.exists():
    code = file_path.read_text(encoding="utf-8")
    updated = re.sub(r"\bnp\.float\b", "float", code)
    file_path.write_text(updated, encoding="utf-8")
    print("✅ 修复完成：已将 np.float 替换为 float")
else:
    print("❌ 错误：文件不存在")

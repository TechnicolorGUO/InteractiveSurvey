import subprocess
import os

def tex_to_pdf(tex_path, output_dir=None, compiler="pdflatex"):
    """
    将 .tex 文件编译为 .pdf 文件。

    参数:
        tex_path (str): 输入的 .tex 文件路径。
        output_dir (str, 可选): 输出的 PDF 存放目录，默认为 .tex 文件所在目录。
        compiler (str, 可选): 使用的 LaTeX 编译器 (pdflatex, xelatex, lualatex)。

    返回:
        str: 生成的 PDF 文件路径，如果失败则返回 None。
    """
    if not os.path.isfile(tex_path):
        print("错误: TeX 文件不存在。")
        return None

    tex_dir = os.path.dirname(tex_path) if output_dir is None else output_dir
    tex_filename = os.path.basename(tex_path)
    pdf_filename = os.path.splitext(tex_filename)[0] + ".pdf"
    pdf_path = os.path.join(tex_dir, pdf_filename)

    # 运行 LaTeX 编译命令
    subprocess.run(
        [compiler, "-interaction=nonstopmode", "-output-directory", tex_dir, tex_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if os.path.isfile(pdf_path):
        print(f"成功: PDF 生成于 {pdf_path}")
        return pdf_path
    else:
        print("错误: PDF 生成失败。")
        return None
    print("错误: LaTeX 编译失败。")
    print(e.stderr.decode("utf-8"))
    return None

# 示例调用
tex_to_pdf(tex_path = "src/static/data/info/undefined/test.tex", output_dir = 'src/static/data/info/undefined', compiler="xelatex")
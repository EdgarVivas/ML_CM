import os
import subprocess
import shutil
from pathlib import Path

def escape_latex(s):
    """
    Escapes LaTeX special characters in a string.

    Parameters:
        s (str): The input string.

    Returns:
        str: The escaped string.
    """
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    for key, val in replacements.items():
        s = s.replace(key, val)
    return s

def generate_latex_document(code_content, language, filename):
    """
    Generates a LaTeX document string embedding the code using lstlisting.

    Parameters:
        code_content (str): The source code to embed.
        language (str): Programming language ('Python' or 'Matlab').
        filename (str): The name of the source file.

    Returns:
        str: The complete LaTeX document as a string.
    """
    # Escape LaTeX special characters
    escaped_code = escape_latex(code_content)

    # Define the language for lstlisting
    if language.lower() == 'python':
        lst_language = 'Python'
    elif language.lower() == 'matlab':
        lst_language = 'Matlab'
    else:
        lst_language = 'text'  # Fallback

    latex_template = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{listings}
\usepackage{color}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{fancyvrb}

\geometry{a4paper, margin=1in}

% Define colors for listings
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegray},
    keywordstyle=\color{codepurple},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\small,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}

\lstset{style=mystyle}

\title{Code Listing: """ + filename + r"""}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Source Code}

\begin{lstlisting}[language=""" + lst_language + r"""]
""" + escaped_code + r"""
\end{lstlisting}

\end{document}
"""
    return latex_template

def convert_code_to_pdf(code_file, output_pdf):
    """
    Converts a code file to a PDF using LaTeX's lstlisting.

    Parameters:
        code_file (Path): Path object to the code file.
        output_pdf (Path): Path object where the PDF will be saved.
    """
    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        print(f"Error reading {code_file}: {e}")
        return

    # Determine language based on file extension
    if code_file.suffix.lower() == '.py':
        language = 'Python'
    elif code_file.suffix.lower() == '.m':
        language = 'Matlab'
    else:
        language = 'text'  # Fallback

    # Generate LaTeX document
    latex_content = generate_latex_document(code_content, language, code_file.name)

    # Temporary LaTeX file
    temp_tex = code_file.with_suffix('.tex')

    try:
        with open(temp_tex, 'w', encoding='utf-8') as f:
            f.write(latex_content)
    except Exception as e:
        print(f"Error writing LaTeX file for {code_file}: {e}")
        return

    # Compile LaTeX to PDF
    try:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', temp_tex.name],
                       cwd=temp_tex.parent,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        print(f"Compiled {code_file.name} to PDF.")
    except subprocess.CalledProcessError:
        print(f"Error compiling LaTeX for {code_file.name}.")
        return

    # Move the generated PDF to the desired output location
    generated_pdf = code_file.with_suffix('.pdf')
    try:
        shutil.move(str(generated_pdf), str(output_pdf))
        print(f"Saved PDF to {output_pdf}")
    except Exception as e:
        print(f"Error moving PDF for {code_file.name}: {e}")

    # Clean up auxiliary files
    for ext in ['.aux', '.log']:
        aux_file = code_file.with_suffix(ext)
        if aux_file.exists():
            aux_file.unlink()

    # Remove the temporary LaTeX file
    if temp_tex.exists():
        temp_tex.unlink()

def process_folder(input_folder, output_folder):
    """
    Processes all .py and .m files in the input folder and converts them to PDFs.

    Parameters:
        input_folder (str): Path to the input folder containing code files.
        output_folder (str): Path to the output folder where PDFs will be saved.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.is_dir():
        print(f"Input folder '{input_folder}' does not exist or is not a directory.")
        return

    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .py and .m files in the input folder (recursively)
    code_files = list(input_path.rglob('*.py')) + list(input_path.rglob('*.m'))

    if not code_files:
        print(f"No Python (.py) or MATLAB (.m) files found in '{input_folder}'.")
        return

    for code_file in code_files:
        # Define output PDF path with the same base name
        pdf_filename = code_file.stem + '.pdf'
        output_pdf = output_path / pdf_filename

        print(f"Processing {code_file}...")
        convert_code_to_pdf(code_file, output_pdf)

    print("All eligible code files have been processed.")

def main():
    """
    Main function to execute the script.
    """
    # Specify the input and output folders here
    input_folder = 'C:/Users/34626/Documents/ML_exam'    # Replace with your input folder path
    output_folder = 'C:/Users/34626/Documents/pdffiles'  # Replace with your output folder path

    # Convert to absolute paths
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")

    process_folder(input_folder, output_folder)
    print("Conversion process completed.")

if __name__ == '__main__':
    main()

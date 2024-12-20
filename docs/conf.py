# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import re
import toml
sys.path.insert(0, os.path.abspath('../src'))  # Source code dir relative to this file

# ---Project Functions----------------------------------------------------

def get_project_info():
    # Determine the directory where the conf.py file is located
    conf_dir = os.path.dirname(__file__)
    
    # Construct the path to the pyproject.toml file located one level up
    pyproject_path = os.path.join(os.path.dirname(conf_dir), 'pyproject.toml')
    
    try:
        with open(pyproject_path, 'r', encoding='utf-8') as toml_file:
            pyproject_data = toml.load(toml_file)
            
            # Get project name, authors, and version
            project_name = pyproject_data.get('tool', {}).get('poetry', {}).get('name', '')
            project_authors = pyproject_data.get('tool', {}).get('poetry', {}).get('authors', [])
            project_version = pyproject_data.get('tool', {}).get('poetry', {}).get('version', '')
            
            # Ensure authors is a list and join multiple authors into a single string
            if isinstance(project_authors, list):
                project_author = ', '.join(project_authors)
            else:
                project_author = str(project_authors)
            
            return project_name, project_author, project_version
    except FileNotFoundError:
        return '', '', ''


def add_math_directive(app, what, name, obj, options, lines):
    # Regular expression pattern to match LaTeX math expressions enclosed in $$
    math_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)

    # Iterate over lines in the docstring and modify LaTeX math expressions
    new_lines = []
    for line in lines:
        new_line = math_pattern.sub(r'.. math:: \n\n   \1', line)
        new_lines.append(new_line)
        new_lines.append('\n')

    lines[:] = new_lines
    

def modify_docstrings(app, what, name, obj, options, lines):
    # Take care of 'Inline emphasis Warning by adding '\' character to single '*'
    if lines:
        # Process each line in the docstring
        for i, line in enumerate(lines):
            # Add a backslash before single '*' characters
            modified_line = re.sub(r'(?<!\\)(\*)', r'\\\1', line)
            lines[i] = modified_line

def setup(app):
    app.connect('autodoc-process-docstring', add_math_directive)
    app.connect('autodoc-process-docstring', modify_docstrings)


# -- Project information -----------------------------------------------------

# Set the 'project' and 'author' variables
project, author, version = get_project_info()

needs_sphinx = '6.2.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',       # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',   # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',   # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',      # Add a link to the Python source code for classes, functions etc.
    'sphinx.ext.napoleon',      # Allows for Google-style docstrings
    'sphinx.ext.mathjax',       # Formats LaTex code using Jax
    'myst_parser',              # Allows parsing of README.md file
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('http://matplotlib.org/', None),
                       'torch': ('https://pytorch.org/docs/stable/', None),
                       'jax': ('https://jax.readthedocs.io/en/latest/', None)}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autosummary_imported_members = False
autosummary_generate = True         # Turn on sphinx.ext.autosummary
autoclass_content = "both"          # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = True         # Show 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True   # If no docstring, inherit from base class
add_module_names = True             # Keep namespaces in class/method signatures (False)Remove namespaces from class/method signatures
myst_heading_anchors = 2            # Allows implicit linking to header names (X levels) with myst_parser (see https://myst-parser.readthedocs.io/en/latest/syntax/cross-referencing.html)
napoleon_include_init_with_doc = True  # True to list __init___ docstrings separately from the class docstring. False to fall back to Sphinx’s default behavior, which considers the __init___ docstring as part of the class documentation. See 
napoleon_use_admonition_for_examples = False
napoleon_use_ivar = True
# For other napoleon options, see https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Exclusions


# -- Options for HTML output -------------------------------------------------

# Readthedocs theme
# html_theme = "sphinx_rtd_theme"
# html_css_files = ["readthedocs-custom.css"] # Override some CSS settings

# Book Theme
html_theme = "sphinx_book_theme"

# Furo Theme
# html_theme = "furo"
# html_logo = "_static/spectacles-logo.png"
# html_favicon = "_static/spectacles-logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

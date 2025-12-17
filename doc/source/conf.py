# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'COMMET solve'
copyright = '2025, Benjamin Alheit & Mathias Peirlinck & Sid Kumar '
author = 'Benjamin Alheit & Mathias Peirlinck & Sid Kumar '
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

# templates_path = ['_templates']
# exclude_patterns = []

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.imgmath',
    'sphinx_rtd_theme',
    # 'sphinx_design',
    # 'sphinx_togglebutton',
    'sphinxcontrib.video',
    # 'nbsphinx',
    # 'nbsphinx_link'
]


templates_path = ['_templates']
# exclude_patterns = []
exclude_patterns = ['_build']
html_favicon = 'logo-icon.svg'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
        'custom.css'
        ]
# html_baseurl = "https://USERNAME.github.io/REPO/"
html_baseurl = "https://commet-code.github.io/commet_solve/"

# -- Ben additions to configuration -------------------------------------------------
# autoclass_content = 'both'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}


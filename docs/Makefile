# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= --fail-on-warning
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

html: Makefile
	@# treat the examples as source files (to be rendered as HTML)
	@cp -f -r ../examples "$(SOURCEDIR)/"
	@# store all the image files from examples in a dedicated static directory
	@# (sphinx requires this, because otherwise it ignores non-source files)
	@mkdir -p "$(SOURCEDIR)/_static/examples"
	@find "$(SOURCEDIR)/examples" -type f \( -name '*.png' -o -name '*.gif' \) -exec cp '{}' "$(SOURCEDIR)/_static/examples" \;
	@# ensure image links in the example READMEs point to the static-files directory in the site source
	@find "$(SOURCEDIR)/examples" -type f -name '*.md' -exec sed -i -e 's|img src="|img src="/_static/examples/|g' '{}' \;
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

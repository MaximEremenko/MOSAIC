# Plugins

The repository contains a small plugin area for experimental or site-specific
extensions, but plugins are not part of the supported release-candidate surface.

Current release-candidate expectations:

- the canonical smoke path does not require plugins
- the supported run path is still:
  - `python -m core.main <run_parameters.json>`
  - `mosaic <run_parameters.json>`
- plugin development is optional and currently secondary to the documented
  scientific-stage workflow under `core/`

If plugin support becomes part of a later public release, this directory should
gain its own compatibility and packaging guidance. For the current release
candidate, treat it as reserved/advanced functionality rather than a core user
entry surface.

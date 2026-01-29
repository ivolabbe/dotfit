# dotfit TODOs

Saved tasks (created by Copilot):

1. Dedupe Boltzmann math
   - Refactor `calculate_multiplet_emissivities` to call `_boltzmann_ratios_for_group`.
   - Run `pytest` and update docstrings.

2. Consolidate multiplet API
   - Implement `multiplet_ratios(tab, method='auto'|'pyneb'|'boltzmann')`.
   - Provide backward-compatible aliases and deprecation notes.
   - Add unit tests for new behavior.

3. Add parser unit tests
   - Add tests for `read_kurucz_table` and `get_line_nist` using small mocked inputs.
   - Avoid network calls by mocking `astroquery` responses.

4. Prune remaining legacy blocks
   - Identify commented/obsolete blocks in `emission_lines.py`.
   - Remove or mark deprecated; ensure tests pass.

---

Status: all items `not-started`. Use the repo-level todo list and this file for tracking.

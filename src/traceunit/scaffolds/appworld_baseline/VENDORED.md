# Vendored: editable AppWorld seed agent

`agent.py` + `__init__.py` are the worldcalib-free AppWorld seed scaffold, copied
from WorldCalib's `agentic/backends/appworld/` (only these two files — the sibling
`runner.py`/`optimizer.py`/`eval_entry.py`/`data.py` were WorldCalib's own harness
and imported `worldcalib`, so they are intentionally excluded).

`agent.py` is a minimal ReAct code agent exposing `solve(world)`; it is the
editable surface the Candidate Editor evolves. It imports only the standard
library and the `appworld` package (loaded inside the external eval venv), never
`worldcalib`. At evaluation time it runs inside a locked-down Docker sandbox
(`appworld_worker.py`) against the frozen solver model.

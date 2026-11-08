"""The on-policy environment: agents doing desktop tasks while the profiling server
profiles the model they call.

  run_onpolicy  run tasks across desktop VMs (`onpolicy-run`)
  preflight     check a task set before a run (`onpolicy-preflight`)
  desktop_env   desktop VMs and task setup, adapted from OSWorld
  mm_agents     the Holo3 and Qwen agents, adapted from OSWorld
"""

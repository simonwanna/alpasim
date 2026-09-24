# Explicit routes

The closed-loop launcher accepts `--route-file` with `--policy alpamayo1_5`.
The file must reside under `--project`. Its JSON schema is:

```json
{
  "scene_id": "selected-scene-id",
  "coordinate_frame": "local",
  "units": "metres",
  "waypoints": [[0, 0, 0], [20, 0, 0], [25, 5, 0], [25, 40, 0]]
}
```

These coordinates illustrate the format only; they are not a validated road
route. Use the selected scene's native local XYZ coordinates, not renderer/NRE
coordinates or image pixels. Build a smooth, lane-aligned path from the approach,
through the intended intersection exit, to the destination. Plot it against the
scene map before running a policy experiment.

The launcher validates the scene identity, numeric coordinates and native route
geometry before starting GPU services. It saves the supplied route as
`run/route.json` and embeds it in `launch.json` and the runtime configuration.
`CUSTOM` mode replaces the recorded route and does not extend past the final
waypoint. With no route file, the launcher retains its recorded-route behavior.

This supplies a route, not an automatic destination-to-route planner. Alpamayo's
existing adapter derives navigation text from the route; the policy still chooses
its trajectory each cycle. The final waypoint does not command a stop, enforce
route following, or automatically terminate the simulation. Verify the supplied
route, policy guidance, predicted trajectories and actual driven path separately.
The route option does not change the requested step count or run duration.

Before a driving experiment, run `service_entry.py --module check_custom_route`
through the existing core environment. This CPU-only check uses synthetic
straight, left and right routes to verify structured configuration, native
resampling and the navigation text adapter together. It loads no model weights
and does not establish that a policy will follow the route. A real scene route
must still be selected and verified on its map.

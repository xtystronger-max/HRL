def test_import_package():
    import hrl3d
    from hrl3d import cli
    from hrl3d.cluster import probe, grouping, loader
    from hrl3d.model import introspect
    from hrl3d.planner_hrl import plan_builder, obs, spaces

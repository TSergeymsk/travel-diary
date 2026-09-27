DEFAULTS_PROCESSING = {
    "vision_max_side": 768,
    "export_max_side": 1600,
    "cluster_radius_m": 400,
    "max_images_per_cluster": 4,
    "use_reverse_geocode": True,

    # ---- метаданные ----
    "use_file_mtime_fallback": True,
    "photo_tz_offset_hours": 0.0,

    # ---- дедупликация ----
    "max_photos_per_day": 6,
    "max_photos_per_location": 4,
    "dedup_time_window_s": 5,
    "dedup_hash_threshold": 10,
    "max_photos_per_cluster": 2,

    # ---- значимость локаций ----
    "significant_min_photos": 3,
    "significant_require_wiki": False,
    "day_intro_min_locations": 2,
    "generate_route_overview": True,

    # ---- GPS ----
    "gps_interpolation_from_gpx": True,
    "gps_interpolation_max_gap_min": 30,
}
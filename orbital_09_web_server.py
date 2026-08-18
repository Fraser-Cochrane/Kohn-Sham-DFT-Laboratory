# =============================================================================
# PART 9 OF 9: FLASK APPLICATION, ROUTES, LAZY-TAB API, AND LOCAL LAUNCHER
# =============================================================================
#
# Defines the Flask app and request lock, validates form submissions, invokes
# the calculation pipeline, serves cached results, and exposes the local launch
# function.  The master script calls launch_web_application() only when run.
#
from __future__ import annotations

app = Flask(__name__)

# The numerical routines currently read a small module-level request context.
# This lock makes threaded local use deterministic and serializes cache writes.
# A production deployment should therefore begin with one WSGI worker; scaling
# to several worker processes requires request-local settings plus a process-safe
# cache lock or background-job queue.
CALCULATION_LOCK = threading.Lock()


@app.get("/")
def home():
    calibration, samples = laptop_runtime_calibration()
    return render_template_string(
        HOME_TEMPLATE,
        tiles=json.dumps(periodic_table_tiles()),
        quality_profiles=json.dumps(QUALITY_PROFILES, separators=(",", ":")),
        runtime_calibration=f"{calibration:.8f}",
        calibration_samples=samples,
    )


@app.post("/calculate")
def calculate_route():
    global ION, SPIN, N_QUANTUM, L_QUANTUM, M_QUANTUM
    global ISOTOPE_MASS_NUMBER
    try:
        symbol = request.form["symbol"].capitalize()
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError("The submitted element symbol is not recognised.")

        charge = int(request.form["charge"])
        electron_count = ATOMIC_NUMBERS[symbol] - charge
        if electron_count < 1:
            raise ValueError("The selected positive charge removes every electron.")

        selected_spin = int(request.form["spin"])
        if selected_spin not in physically_allowed_spin_values(electron_count):
            raise ValueError(
                f"For {electron_count} electrons, 2S must have parity "
                f"{electron_count % 2} and satisfy 0 <= 2S <= {electron_count}."
            )

        selected_n = int(request.form["n"])
        selected_l = int(request.form["l"])
        selected_m = int(request.form["m"])
        if selected_n < 1 or not 0 <= selected_l < selected_n:
            raise ValueError("Quantum numbers must satisfy n >= 1 and 0 <= l < n.")
        if selected_n > COMMON_ORBITAL_MAX_N:
            raise ValueError(
                f"The fast website supports n <= {COMMON_ORBITAL_MAX_N}."
            )
        if abs(selected_m) > selected_l:
            raise ValueError("The magnetic quantum number must satisfy -l <= m <= l.")

        selected_isotope = int(request.form["isotope"])
        if selected_isotope < ATOMIC_NUMBERS[symbol]:
            raise ValueError("The isotope mass number A cannot be smaller than Z.")

        selected_quality = int(request.form["quality"])
        if selected_quality not in QUALITY_PROFILES:
            raise ValueError("Accuracy level must be an integer from 1 to 5.")

        selected_ion = (
            symbol
            if charge == 0
            else f"{symbol}{abs(charge)}{'+' if charge > 0 else '-'}"
        )
        # Validation happens outside the lock; only the shared numerical context
        # and calculation are serialized. Cached results therefore return
        # quickly while two cold calculations cannot mix their global settings.
        with CALCULATION_LOCK:
            # The numerical routines use these settings as a single immutable
            # request context.  Assigning them inside the lock prevents two
            # simultaneous browser requests from mixing quantum numbers.
            ION = selected_ion
            SPIN = selected_spin
            N_QUANTUM = selected_n
            L_QUANTUM = selected_l
            M_QUANTUM = selected_m
            ISOTOPE_MASS_NUMBER = selected_isotope
            apply_quality_level(selected_quality)

            # Only genuinely uncached calculations calibrate the laptop model;
            # warm/cached timings would otherwise make future predictions far
            # too optimistic.
            prospective_atomic_key = dft_cache_key(symbol, charge)
            prospective_visual_key = result_cache_key(prospective_atomic_key)
            atomic_path = cache_bundle("dft-fields", prospective_atomic_key)
            result_path = cache_file(
                "rendered-results", prospective_visual_key, ".html"
            )
            try:
                atomic_cached = load_array_bundle(atomic_path, ()) is not None
                result_cached = (
                    result_path is not None
                    and result_path.is_file()
                    and result_path.stat().st_size >= RESULT_CACHE_MIN_BYTES
                    and representation_cache_ready(prospective_visual_key)
                )
            except OSError:
                atomic_cached = False
                result_cached = False
            calculation_start = time.perf_counter()
            result = calculate_current_selection()
            elapsed = time.perf_counter() - calculation_start
            record_runtime_sample(
                atomic_number=ATOMIC_NUMBERS[symbol],
                electron_count=electron_count,
                spin_2s=selected_spin,
                ionic_charge=charge,
                quality_level=selected_quality,
                seconds=elapsed,
                cold_run=not atomic_cached and not result_cached,
            )
    except Exception as exc:
        return (
            render_template_string(
                ERROR_TEMPLATE,
                error_type=type(exc).__name__,
                message=str(exc),
            ),
            400,
        )
    result_url = url_for("result_file", filename=result.name)
    if request.headers.get("X-Orbital-Progress") == "1":
        return Response(
            json.dumps({"result_url": result_url}),
            mimetype="application/json",
        )
    return redirect(result_url)


@app.get("/api/result-tabs/<visual_key>/<panel_id>")
def lazy_result_tab(visual_key: str, panel_id: str):
    """Generate one result tab from cached wavefunction data on first use.

    The result page initially contains only its first figure. Opening another
    tab calls this endpoint, which restores the saved request context, rebuilds
    only that representation from physical caches, stores Plotly JSON, and then
    restores the process context in ``finally``.
    """
    global ION, SPIN, ORBITAL_SPIN, N_QUANTUM, L_QUANTUM, M_QUANTUM
    global ORBITAL_FORM, ISOTOPE_MASS_NUMBER

    if re.fullmatch(r"[0-9a-f]{64}", visual_key) is None:
        return Response(
            json.dumps({"error": "Invalid result identifier."}),
            status=400,
            mimetype="application/json",
        )
    if panel_id not in VIEW_DEFINITIONS:
        return Response(
            json.dumps({"error": "Unknown visualization tab."}),
            status=404,
            mimetype="application/json",
        )

    figure_path = lazy_figure_cache_path(visual_key, panel_id)
    cached_json = read_cached_figure_json(figure_path)
    if cached_json is not None:
        response = Response(cached_json, mimetype="application/json")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    try:
        data, metadata = load_wavefunction_data(visual_key, panel_id)
        with CALCULATION_LOCK:
            # A second request may have completed the same tab while this one
            # waited for the numerical/rendering lock.
            cached_json = read_cached_figure_json(figure_path)
            if cached_json is None:
                previous_context = (
                    ION,
                    SPIN,
                    ORBITAL_SPIN,
                    N_QUANTUM,
                    L_QUANTUM,
                    M_QUANTUM,
                    ORBITAL_FORM,
                    ISOTOPE_MASS_NUMBER,
                    QUALITY_LEVEL,
                )
                try:
                    ION = str(metadata["ion"])
                    SPIN = int(metadata["spin_2s"])
                    ORBITAL_SPIN = str(metadata["orbital_spin"])
                    N_QUANTUM = int(metadata["n"])
                    L_QUANTUM = int(metadata["l"])
                    M_QUANTUM = int(metadata["m"])
                    ORBITAL_FORM = str(metadata["form"])
                    isotope = metadata.get("isotope")
                    ISOTOPE_MASS_NUMBER = None if isotope is None else int(isotope)
                    apply_quality_level(int(metadata["quality_level"]))
                    label = VIEW_DEFINITIONS[panel_id][0]
                    figure = build_figure_safely(
                        label,
                        lambda: build_lazy_figure_from_data(
                            panel_id, data, metadata
                        ),
                    )
                    cached_json = figure.to_json()
                    atomic_write_text(figure_path, cached_json)
                    prune_cache(exclude=figure_path)
                finally:
                    (
                        ION,
                        SPIN,
                        ORBITAL_SPIN,
                        N_QUANTUM,
                        L_QUANTUM,
                        M_QUANTUM,
                        ORBITAL_FORM,
                        ISOTOPE_MASS_NUMBER,
                        previous_quality,
                    ) = previous_context
                    apply_quality_level(previous_quality)
        response = Response(cached_json, mimetype="application/json")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
    except Exception as exc:
        print(
            f"Lazy tab {panel_id} failed safely "
            f"({type(exc).__name__}: {exc})."
        )
        return Response(
            json.dumps(
                {"error": f"{VIEW_DEFINITIONS[panel_id][0]} could not be generated."}
            ),
            status=500,
            mimetype="application/json",
        )


@app.get("/results/<path:filename>")
def result_file(filename: str):
    root = cache_root()
    directory = root / "rendered-results" if root is not None else Path.cwd()
    if not (directory / filename).is_file():
        directory = Path.cwd()
    return send_from_directory(directory, filename)


def launch_web_application() -> None:
    """Start the local development UI and open it in the default browser.

    Public hosting must import ``app`` with a production WSGI server instead;
    the ``if __name__`` guard ensures Gunicorn does not launch a browser or the
    Flask development server when it imports this module.
    """
    url = f"http://{WEB_HOST}:{WEB_PORT}/"
    print(f"Atomic Orbital Explorer: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)

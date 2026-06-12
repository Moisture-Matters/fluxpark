import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import fluxpark as flp
from fluxpark.submodels.interception import interception_voortman
from fluxpark.submodels.rootwateruptake import unsat_reservoirmodel
from fluxpark.submodels.soilevaporation import soilevap_boestenstroosnijder
from fluxpark.workflow import ports

logger = logging.getLogger(__name__)


class FluxParkRunner:
    """
    Runner class for executing a FluxPark simulation.

    Parameters
    ----------
    cfg_core : FluxParkConfig
        Core configuration for the FluxPark model.
    runner_ports : RunnerPorts, optional
        Collection of workflow ports. If omitted, default open-source
        ports are used.
    """

    def __init__(
        self,
        cfg_core: flp.config.FluxParkConfig,
        runner_ports: Optional[ports.RunnerPorts] = None,
    ):
        self.cfg = cfg_core
        self.ports = runner_ports or ports.RunnerPorts()

        # runtime/setup state
        self.context: Dict[str, Any] = {}
        self.initial_data: Dict[str, Any] = {}
        self.ancillary_rasters: Dict[str, Any] = {}

        # daily runtime state
        self.current_input_rasters: Dict[str, Any] = {}
        self.current_evap_params: Dict[str, Any] = {}

    def setup(self) -> None:
        """Prepare static data and initial model state."""
        cfg = self.cfg

        # runtime/environment-specific context
        self.context = self.ports.execution_context(cfg)

        # time information
        self.dates = flp.setup.parse_dates(cfg.date_start, cfg.date_end)

        # directories
        (
            self.outdir,
            self.indir_tables,
            self.indir_rasters,
            self.indir_masks,
            self.intermediate_dir,
        ) = flp.setup.resolve_dirs(
            cfg.outdir,
            cfg.indir,
            cfg.indir_tables,
            cfg.indir_rasters,
            cfg.indir_masks,
            cfg.intermediate_dir,
        )

        # grid parameters
        self.grid_params = flp.setup.compute_grid_params(
            cfg.x_min,
            cfg.x_max,
            cfg.y_min,
            cfg.y_max,
            cfg.cellsize,
            cfg.calc_epsg_code,
            self.indir_masks,
            cfg.mask,
        )

        # evaporation parameters
        self.evap_params = flp.setup.load_evap_params(
            self.indir_tables,
            cfg.evap_param_table,
        )

        # land-use / evaporation conversion
        self.luse_ids, self.evap_ids, self.luse_label = flp.setup.load_luse_evap_conv(
            self.indir_tables
        )

        # output conversion
        self.conv_output_table, self.conv_output, self.conv_var = (
            flp.setup.load_conv_output(
                self.indir_tables,
                cfg.output_mapping,
            )
        )

        # active modules and raster names
        self.mods = {
            key: value for key, value in vars(cfg).items() if key.startswith("mod_")
        }
        self.rasternames = {
            key: value
            for key, value in vars(cfg).items()
            if key.endswith("_rastername")
        }

        # output and rerun bookkeeping
        (
            self.out_par_list,
            self.calc_par_list,
            self.cum_par_list,
            self.out_var_list,
            self.rerun_par_list,
            self.rerun_var_list,
        ) = flp.setup.prepare_output_and_rerun_lists(
            self.mods,
            cfg.output_files,
            self.conv_output_table,
            self.conv_output,
            cfg.store_states,
            cfg.eval_waterbalance,
        )

        # dynamic land use
        self.dynamic_landuse, self.input_raster_years = (
            flp.setup.detect_dynamic_landuse_and_years(
                cfg.landuse_rastername,
                cfg.root_soilm_scp_rastername,
                cfg.root_soilm_pwp_rastername,
                self.indir_rasters,
            )
        )

        # static input rasters
        self.soil_cov_decid, self.soil_cov_conif = flp.setup.read_static_maps(
            self.indir_rasters,
            self.grid_params,
            self.mods,
            cfg.soil_cov_decid_rastername,
            cfg.soil_cov_conif_rastername,
        )

        # initial loop state
        self.old = flp.setup.init_old(
            self.rerun_var_list,
            self.grid_params["nrows"],
            self.grid_params["ncols"],
        )

        # optional initial data/state override
        self.initial_data = self.ports.initial_data(self)
        self.old.update(self.initial_data.get("old", {}))

    def _log_timing_summary(self, timings: Dict[str, float], total_time: float) -> None:
        """Log timing summary for the run."""
        total_stages = sum(timings.values())
        overhead = total_time - total_stages

        logger.info("finished calculations %.2f sec", total_time)

        for label, value in timings.items():
            pct = (value / total_time * 100.0) if total_time > 0 else 0.0
            logger.info("%s %.2f sec, %.2f %%", label, value, pct)

        overhead_pct = (overhead / total_time * 100.0) if total_time > 0 else 0.0
        logger.info(
            "initialization and overhead %.2f sec, %.2f %%",
            overhead,
            overhead_pct,
        )

    def run(self) -> None:
        """Execute the FluxPark simulation."""
        self.setup()
        cfg = self.cfg
        old = self.old

        flp.config.save_cfg(cfg, self.outdir)

        if cfg.eval_waterbalance:
            init_date = self.dates[0] - pd.Timedelta(days=1)
            nrows = self.grid_params["nrows"]
            ncols = self.grid_params["ncols"]
            flp.postprocessing.write_output_tif(
                self.old.get("smda", np.zeros((nrows, ncols), dtype=np.float32)).copy(),
                f"{init_date.strftime('%Y%m%d')}-soilm_def_act_mm.tif",
                np.ones((nrows, ncols), dtype=np.int32),
                [],
                False,
                self.outdir,
                cfg.x_min, cfg.y_max, cfg.cellsize, cfg.calc_epsg_code,
            )

        total_start = time.time()

        tot_time_rain_prep = 0.0
        tot_time_etref_prep = 0.0
        tot_time_raster_prep = 0.0
        tot_time_evappar_prep = 0.0
        tot_time_int_calc = 0.0
        tot_time_soilevap_calc = 0.0
        tot_time_trans_calc = 0.0
        tot_time_postp = 0.0
        tot_time_writing = 0.0

        # Explicitly initialize for typing and yearly refresh logic.
        landuse_map: Optional[NDArray[Any]] = None
        soilm_scp: Optional[NDArray[Any]] = None
        soilm_pwp: Optional[NDArray[Any]] = None
        imperv: Optional[NDArray[Any]] = None
        beta: Optional[NDArray[Any]] = None

        for i, date in enumerate(self.dates):
            self.i = i
            self.date = date
            self.is_new_year = date.day == 1 and date.month == 1

            logger.info("t = %s", date.date())

            daily_output: Dict[str, Any] = {}

            # prepare input rasters
            start_time_raster_prep = time.time()

            if self.is_new_year or i == 0:
                (
                    landuse_map,
                    soilm_scp,
                    soilm_pwp,
                    imperv,
                    beta,
                ) = flp.prepgrids.load_fluxpark_raster_inputs(
                    date=date,
                    indir_rasters=self.indir_rasters,
                    grid_params=self.grid_params,
                    dynamic_landuse=self.dynamic_landuse,
                    landuse_filename=cfg.landuse_rastername,
                    root_soilm_scp_filename=cfg.root_soilm_scp_rastername,
                    root_soilm_pwp_filename=cfg.root_soilm_pwp_rastername,
                    impervdens_filename=cfg.impervdens_rastername,
                    input_raster_years=self.input_raster_years,
                    luse_ids=self.luse_ids,
                    bare_soil_ids=cfg.bare_soil_ids,
                    urban_ids=cfg.urban_ids,
                )

                # Prevent NaN values in old["smda"] from propagating into new maps.
                old["smda"] = np.where(np.isnan(old["smda"]), 0, old["smda"])

            assert landuse_map is not None, "landuse_map must be defined"
            assert soilm_scp is not None, "soilm_scp must be defined"
            assert soilm_pwp is not None, "soilm_pwp must be defined"
            assert imperv is not None, "imperv must be defined"
            assert beta is not None, "beta must be defined"

            self.current_input_rasters = {
                "landuse_map": landuse_map,
                "soilm_scp": soilm_scp,
                "soilm_pwp": soilm_pwp,
                "imperv": imperv,
                "beta": beta,
            }

            self.ancillary_rasters = self.ports.ancillary_raster(self)
            self.current_input_rasters.update(self.ancillary_rasters)

            tot_time_raster_prep += time.time() - start_time_raster_prep

            # prepare evaporation parameters
            start_time_evappar_prep = time.time()
            self.current_evap_params = flp.prepgrids.apply_evaporation_parameters(
                self.luse_ids,
                self.evap_ids,
                self.evap_params,
                date.dayofyear,
                landuse_map,
                imperv,
                cfg.urban_ids,
                mod_vegcover=cfg.mod_vegcover,
                soil_cov_decid=self.soil_cov_decid,
                soil_cov_conif=self.soil_cov_conif,
            )
            tot_time_evappar_prep += time.time() - start_time_evappar_prep

            # rain input
            start_time_rain_prep = time.time()
            rain = self.ports.rain_provider(self)
            skip_day = flp.utils.validate_grid(
                rain,
                expected_shape=(
                    self.grid_params["nrows"],
                    self.grid_params["ncols"],
                ),
                name="rain",
                nan_policy=cfg.nan_policy,
            )
            if skip_day:
                logger.warning("Skipping day because rain contains NaN or is None.")
                continue
            tot_time_rain_prep += time.time() - start_time_rain_prep

            # etref input
            start_time_etref_prep = time.time()
            etref = self.ports.etref_provider(self)
            skip_day = flp.utils.validate_grid(
                etref,
                expected_shape=(
                    self.grid_params["nrows"],
                    self.grid_params["ncols"],
                ),
                name="etref",
                nan_policy=cfg.nan_policy,
            )
            if skip_day:
                logger.warning("Skipping day because etref contains NaN or is None.")
                continue
            tot_time_etref_prep += time.time() - start_time_etref_prep

            self.current_evap_params.update(
                {
                    "rain": rain,
                    "etref": etref,
                }
            )

            # pre-core modifier
            modifier = self.ports.daily_input_modifier(self, old)
            daily_output.update(modifier["daily_output"])
            old.update(modifier["states"])
            self.current_evap_params.update(modifier["current_evap_params"])

            # unpack current evaporation parameters
            evap = self.current_evap_params
            trans_fact = evap["trans_fact"]
            soil_evap_fact = evap["soil_evap_fact"]
            int_cap = evap["int_cap"]
            soil_cov = evap["soil_cov"]
            openwater_fact = evap["openwater_fact"]
            rain = evap["rain"]
            etref = evap["etref"]

            # interception
            start_time_int_calc = time.time()
            int_evap, int_store, throughfall, int_timefrac = interception_voortman(
                etref * 1.25 + int_cap,
                rain,
                int_cap,
                soil_cov,
                old["int_store"],
            )
            tot_time_int_calc += time.time() - start_time_int_calc

            # soil evaporation
            start_time_soilevap_calc = time.time()
            soil_evap_pot = etref * soil_evap_fact * (1.0 - soil_cov)
            soil_evap_act_est, sum_ep, sum_ea = soilevap_boestenstroosnijder(
                throughfall,
                soil_evap_pot,
                beta,
                old["sum_ep"],
                old["sum_ea"],
            )
            tot_time_soilevap_calc += time.time() - start_time_soilevap_calc

            # transpiration
            start_time_trans_calc = time.time()
            trans_pot = etref * trans_fact * soil_cov * (1.0 - int_timefrac)

            valid = soilm_pwp != -9999
            mask = valid & (old["smda"] > soilm_pwp)
            old["smda"][mask] = soilm_pwp[mask]

            # open water evaporation
            open_water_evap_act = openwater_fact * etref

            # reservoir model
            eta, smdp, smda, prec_surplus = unsat_reservoirmodel(
                throughfall,
                soil_evap_act_est + trans_pot,
                old["smda"],
                soilm_scp,
                soilm_pwp,
            )

            tot_time_trans_calc += time.time() - start_time_trans_calc

            # post-process daily rasters
            start_time_postp = time.time()

            daily_output_update = flp.postprocessing.post_process_daily(
                eta,
                trans_pot,
                soil_evap_act_est,
                int_evap,
                soil_evap_pot,
                open_water_evap_act,
                smda,
                soilm_pwp,
                rain,
                etref,
                landuse_map,
                prec_surplus,
                cfg.open_water_ids,
            )

            # runoff for impervious area
            runoff = prec_surplus * imperv * cfg.impervious_runoff_fraction

            # recharge
            recharge = prec_surplus - runoff

            daily_output.update(daily_output_update)
            daily_output.update(
                {
                    "rain": rain,
                    "etref": etref,
                    "throughfall": throughfall,
                    "int_store": int_store,
                    "sum_ep": sum_ep,
                    "sum_ea": sum_ea,
                    "runoff": runoff,
                    "recharge": recharge,
                }
            )

            # post-core modifier
            modifier = self.ports.daily_output_modifier(daily_output, old, self)
            daily_output.update(modifier["daily_output"])
            old.update(modifier["states"])

            # cumulative outputs
            cum_output, old = flp.postprocessing.update_cumulative_fluxes(
                daily_output,
                old,
                date,
                cfg.reset_cum_day,
                cfg.reset_cum_month,
                self.cum_par_list,
                self.conv_output,
            )

            flp.workflow.update_loop_state(
                old,
                self.rerun_par_list,
                self.conv_output,
                daily_output,
                cum_output,
            )
            tot_time_postp += time.time() - start_time_postp

            # write output
            start_time_writing = time.time()
            flp.postprocessing.write_all_tiffs(
                date,
                self.out_par_list,
                self.conv_output,
                daily_output,
                cum_output,
                landuse_map,
                cfg.write_nan_for_landuse_ids,
                cfg.replace_nan_with_zero,
                self.outdir,
                cfg.x_min,
                cfg.y_max,
                cfg.cellsize,
                cfg.calc_epsg_code,
                cfg.only_yearly_output,
                cfg.parallel,
                cfg.max_workers,
            )
            tot_time_writing += time.time() - start_time_writing

            # optional output hook
            self.ports.output(self)

        total_time = time.time() - total_start
        timings = {
            "preparing rain": tot_time_rain_prep,
            "preparing etref": tot_time_etref_prep,
            "preparing input rasters": tot_time_raster_prep,
            "preparing evaporation parameters": tot_time_evappar_prep,
            "calculation interception": tot_time_int_calc,
            "calculation soil evaporation": tot_time_soilevap_calc,
            "calculation transpiration": tot_time_trans_calc,
            "post-processing": tot_time_postp,
            "writing output": tot_time_writing,
        }
        self._log_timing_summary(timings, total_time)

        if cfg.eval_waterbalance:
            flp.postprocessing.eval_waterbalance(self.outdir)

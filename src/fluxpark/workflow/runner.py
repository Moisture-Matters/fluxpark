import time
import logging
import sys
import numpy as np
from numpy.typing import NDArray
import fluxpark as flp
from fluxpark.submodels.interception import interception_voortman
from fluxpark.submodels.soilevaporation import soilevap_boestenstroosnijder
from fluxpark.submodels.rootwateruptake import unsat_reservoirmodel

from typing import Optional, Dict, Any


class FluxParkRunner:
    """
    Runner class for executing a FluxPark simulation using a provided configuration.

    Parameters
    ----------
    cfg_core : FluxParkConfig
        Core configuration for the FluxPark model.
    """

    def __init__(
        self,
        cfg_core: flp.config.FluxParkConfig,
        input_hooks: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg_core
        self.input_hooks = input_hooks or {}
        self.setup_logger()

    def setup_logger(self):
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def default_rain_input(self, date, grid_params):
        shape = (grid_params["nrows"], grid_params["ncols"])
        rain = np.full(shape, 1.0, dtype="float32")
        return rain

    def default_etref_input(self, date, grid_params):
        shape = (grid_params["nrows"], grid_params["ncols"])
        etref = np.full(shape, 1.0, dtype="float32")
        return etref

    def setup(self):
        cfg = self.cfg

        # time information
        self.dates = flp.setup.parse_dates(cfg.date_start, cfg.date_end)

        # directories
        (
            self.outdir,
            self.indir,
            self.indir_rasters,
            self.indir_masks,
            self.intermediate_dir,
        ) = flp.setup.resolve_dirs(
            cfg.outdir,
            cfg.indir,
            cfg.indir_rasters,
            cfg.indir_masks,
            cfg.intermediate_dir,
        )

        # compute grid parameters
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

        # read evaporation parameters
        self.evap_params = flp.setup.load_evap_params(self.indir, cfg.evap_param_table)

        # read the ids of landuse and corresponding evaporation ids
        (self.luse_ids, self.evap_ids, self.luse_label) = flp.setup.load_luse_evap_conv(
            self.indir
        )

        # read conversion table from luse
        (self.conv_output_table, self.conv_output, self.conv_var) = (
            flp.setup.load_conv_output(self.indir, cfg.output_mapping)
        )

        # merge active mods and raster names in a dict.
        self.mods = {k: v for k, v in vars(cfg).items() if k.startswith("mod_")}
        self.rasternames = {
            k: v for k, v in vars(cfg).items() if k.endswith("_rastername")
        }

        # determine all output names, variable names, and type
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
        )

        # initialize rasters of the previous timestep (initial)
        self.old = flp.setup.init_old(
            self.rerun_var_list, self.grid_params["nrows"], self.grid_params["ncols"]
        )

        # determine the use of dynamic landuse and the provided years.
        (self.dynamic_landuse, self.input_raster_years) = (
            flp.setup.detect_dynamic_landuse_and_years(
                cfg.landuse_rastername,
                cfg.root_soilm_scp_rastername,
                cfg.root_soilm_pwp_rastername,
                self.indir_rasters,
            )
        )

        # read static input rasters
        (self.imperv, self.soil_cov_conif, self.soil_cov_decid) = (
            flp.setup.read_static_maps(
                self.indir_rasters,
                self.grid_params,
                self.mods,
                cfg.impervdens_rastername,
                cfg.soil_cov_decid_rastername,
                cfg.soil_cov_conif_rastername,
            )
        )

    def run(self):
        self.setup()
        cfg = self.cfg
        old = self.old

        tot_time = time.time()
        tot_time_rain_prep = 0.0
        tot_time_etref_prep = 0.0
        tot_time_veg_prep = 0.0
        tot_time_mod_seen = 0.0
        tot_time_int_calc = 0.0
        tot_time_soilevap_calc = 0.0
        tot_time_trans_calc = 0.0
        tot_time_postp = 0.0
        tot_time_writing = 0.0

        # initialize array to explitly test typing.
        landuse_map: Optional[NDArray[Any]] = None
        soilm_scp: Optional[NDArray[Any]] = None
        soilm_pwp: Optional[NDArray[Any]] = None
        beta: Optional[NDArray[Any]] = None
        for i, date in enumerate(self.dates):
            logging.info(f"t = {date.date()}")

            is_new_year = date.day == 1 and date.month == 1
            if is_new_year or i == 0:
                (landuse_map, soilm_scp, soilm_pwp, beta) = (
                    flp.prepgrids.load_fluxpark_raster_inputs(
                        date=date,
                        indir_rasters=self.indir_rasters,
                        grid_params=self.grid_params,
                        dynamic_landuse=self.dynamic_landuse,
                        landuse_filename=cfg.landuse_rastername,
                        root_soilm_scp_filename=cfg.root_soilm_scp_rastername,
                        root_soilm_pwp_filename=cfg.root_soilm_pwp_rastername,
                        input_raster_years=self.input_raster_years,
                        imperv=self.imperv,
                        luse_ids=self.luse_ids,
                    )
                )
            assert landuse_map is not None, "landuse_map must be defined"
            assert soilm_scp is not None, "soilm_scp must be defined"
            assert soilm_pwp is not None, "soilm_pwp must be defined"
            assert beta is not None, "beta must be defined"

            # vegetation data
            start_time_veg_prep = time.time()
            (trans_fact, soil_evap_fact, int_cap, soil_cov, openwater_fact) = (
                flp.prepgrids.apply_evaporation_parameters(
                    self.luse_ids,
                    self.evap_ids,
                    self.evap_params,
                    date.dayofyear,
                    landuse_map,
                    self.imperv,
                    mod_vegcover=cfg.mod_vegcover,
                    soil_cov_decid=self.soil_cov_decid,
                    soil_cov_conif=self.soil_cov_conif,
                )
            )
            tot_time_veg_prep += time.time() - start_time_veg_prep

            # placeholder for rain and etref input
            get_rain = self.input_hooks.get("get_rain", self.default_rain_input)
            rain = get_rain(date, self.grid_params)
            get_etref = self.input_hooks.get("get_etref", self.default_etref_input)
            etref = get_etref(date, self.grid_params)

            # # placeholder for rain and etref input (must be floats)
            # rain = np.full(np.shape(landuse_map), 1.0).astype("float32")
            # etref = np.full(np.shape(landuse_map), 1.0).astype("float32")

            # interception
            start_time_int_calc = time.time()
            int_evap, int_store, throughfall, int_timefrac = interception_voortman(
                etref * 1.25 + int_cap, rain, int_cap, soil_cov, old["int_store"]
            )
            tot_time_int_calc += time.time() - start_time_int_calc

            # soil evaporation
            start_time_soilevap_calc = time.time()
            soil_evap_pot = etref * soil_evap_fact * (1.0 - soil_cov)
            soil_evap_act_est, sum_ep, sum_ea = soilevap_boestenstroosnijder(
                throughfall, soil_evap_pot, beta, old["sum_ep"], old["sum_ea"]
            )
            tot_time_soilevap_calc += time.time() - start_time_soilevap_calc

            # transpiration
            start_time_trans_calc = time.time()
            trans_pot = etref * trans_fact * soil_cov * (1.0 - int_timefrac)
            old["smda"][old["smda"] > soilm_pwp] = soilm_pwp[old["smda"] > soilm_pwp]
            eta, smdp, smda, prec_surplus = unsat_reservoirmodel(
                throughfall,
                soil_evap_act_est + trans_pot,
                old["smda"],
                soilm_scp,
                soilm_pwp,
            )
            open_water_evap_act = openwater_fact * etref
            tot_time_trans_calc += time.time() - start_time_trans_calc

            # post processing
            start_time_postp = time.time()
            daily_output = flp.postprocessing.post_process_daily(
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
            )
            daily_output.update(
                {
                    "rain": rain,
                    "etref": etref,
                    "throughfall": throughfall,
                    "int_store": int_store,
                    "sum_ep": sum_ep,
                    "sum_ea": sum_ea,
                }
            )

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
                old, self.rerun_par_list, self.conv_output, daily_output, cum_output
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

        # logging timings
        tot_time = time.time() - tot_time
        stages = [
            tot_time_rain_prep,
            tot_time_etref_prep,
            tot_time_veg_prep,
            tot_time_mod_seen,
            tot_time_int_calc,
            tot_time_soilevap_calc,
            tot_time_trans_calc,
            tot_time_postp,
            tot_time_writing,
        ]
        tot_time_overhead = tot_time - sum(stages)
        logging.info(f"finished calculations {tot_time:.2f} sec")
        logging.info(
            f"preparing rain {tot_time_rain_prep:.2f} sec, "
            f"{tot_time_rain_prep/tot_time*100:.2f} %"
        )
        logging.info(
            f"preparing etref {tot_time_etref_prep:.2f} sec, "
            f"{tot_time_etref_prep/tot_time*100:.2f} %"
        )
        logging.info(
            f"preparing vegetation data {tot_time_veg_prep:.2f} sec, "
            f"{tot_time_veg_prep/tot_time*100:.2f} %"
        )
        logging.info(
            f"calculation interception {tot_time_int_calc:.2f} sec, "
            f"{tot_time_int_calc/tot_time*100:.2f} %"
        )
        logging.info(
            f"calculation soil evaporation {tot_time_soilevap_calc:.2f} sec, "
            f"{tot_time_soilevap_calc/tot_time*100:.2f} %"
        )
        logging.info(
            f"calculation transpiration {tot_time_trans_calc:.2f} sec, "
            f"{tot_time_trans_calc/tot_time*100:.2f} %"
        )
        logging.info(
            f"post-processing {tot_time_postp:.2f} sec, "
            f"{tot_time_postp/tot_time*100:.2f} %"
        )
        logging.info(
            f"writing output {tot_time_writing:.2f} sec, "
            f"{tot_time_writing/tot_time*100:.2f} %"
        )
        logging.info(
            f"initialization and overhead {tot_time_overhead:.2f} sec, "
            f"{tot_time_overhead/tot_time*100:.2f} %"
        )

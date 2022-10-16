# Outcome Homogenization

This codebase provides tooling to measure outcome homogenization.
Additionally, it provides the code for running experiments and visualizing results described in the NeurIPS 2022 paper **Picking on the Same Person: Does Algorithmic Monoculture lead to Outcome Homogenization?** by Rishi Bommasani, Kathleen A. Creel, Ananya Kumar, Dan Jurafsky, and Percy Liang.


# Roadmap

To measure homogenization, refer to `src/homogenization.py` for the specific measures.
To replicate experiments from the NeurIPS 2022 paper.

- Training data: You will need to download the ACS, LSAC, or GC data locally to `src/data`. Once done, the corresponding bash script will train and evaluate models, with the underlying code specified in the corresponding `.py` file.
- Vision experiments: Models are trained using the code provided in https://github.com/AnanyaKumar/transfer_learning. Once checkpoints are stored in `src/predictions/celeba`, `cv_experiments.py` will handle grouping (to study both individual and group-level homogenization) and measure homogenization. Groupings rely on pre-provided metadata, which is handled by `cv_groupings.py`.
- Language experiments: Models are trained using the code in `train_nlp.py` with configurations specified in `src/conf` and the core modeling training code specified in `adaptation_nlp.py`, which can be launched using the bash script `train_nlp.sh`. Once checkpoints are generated, `nlp_experiments.py` will handle grouping (to study both individual and group-level homogenization) and measure homogenization. Since the language datasets are more heterogeneous, groupings do not rely on pre-provided metadata (in comparison to vision) and instead require grouping of data, which is handled by `nlp_groupings.py`.

To visualize results, all functionality is provided in `visuals.py`. 


# Use cases

The main use case we envision for the codebase is to meausre homogenization on your own data. For this, the main utility is provided by `homogenization.py`, wherein the data structures assumed are pre-specified. We plan to release an even simpler package in the future to facilitate homogenization measurement.

# Questions

Any questions or concerns should be directed to rishibommasani@gmail.com. This codebase is being actively developed upon and improved in followup works (as of October 2022), so it is subject to change. Please refer to the commits cerca October 15, 2022 for directly reproducing the NeurIPS paper.


# Citation

```
@inproceedings{bommasani2022homogenization,
	title     = {Picking on the Same Person: Does Algorithmic Monoculture lead to Outcome Homogenization?},
	author    = {Rishi Bommasani and Kathleen A. Creel and Ananya Kumar and Dan Jurafsky and Percy Liang},
	booktitle = {Advances in Neural Information Processing Systems},
	year      = {2022}
}
```

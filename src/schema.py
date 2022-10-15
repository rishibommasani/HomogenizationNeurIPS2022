from typing import Any, Dict

from quinine.common.cerberus import default, merge, nullable, tboolean, tfloat, tinteger, tlist, tstring

def get_train_schema() -> Dict[str, Any]:
	schema = {
		"seed": merge(tinteger, nullable, default(21)),
		"output_dir": merge(tstring, nullable, default("")),
		"cluster": merge(tboolean, nullable, default(True)),
		"model_name": merge(tstring, nullable, default('roberta-base')),
	}

	return schema
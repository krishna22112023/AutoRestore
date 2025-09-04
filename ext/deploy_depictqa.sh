git clone https://github.com/XPixelGroup/DepictQA.git DepictQA

cp custom_depictqa_scripts/app_eval.py DepictQA/src/
cp custom_depictqa_scripts/app_comp.py DepictQA/src/

mkdir DepictQA/experiments/agenticir
cp custom_depictqa_scripts/config_eval.yaml DepictQA/experiments/agenticir/
cp custom_depictqa_scripts/config_comp.yaml DepictQA/experiments/agenticir/

mkdir DepictQA/weights/delta -p

cp tune_depictqa/* DepictQA/experiments/agenticir

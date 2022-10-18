## KERC Competition Repo of CNUSclab Team
This is repo for [KERC 2022 Competition](http://15.165.135.10/en)

## Training
------
```
python train.py --pretrained <kobert|electr>
```

## Generate submission
------
```
python submit.py --model <model_name>.bin --pretrained <kobert|electr> --output <file_name>.csv --input <private_test_data|public_test_data>.tsv
```
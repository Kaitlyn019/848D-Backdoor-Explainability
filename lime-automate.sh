#!/bin/sh
#118
python explanation/main.py --model_filepath=./models/id-00000118/model.pt --tokenizer_filepath=./tokenizers/GPT-2-gpt2.pt --embedding_filepath=./embeddings/GPT-2-gpt2.pt --examples_dirpath=./models/id-00000118/clean_example_data/class_0_example_1.txt --file_name=clean_118
python explanation/main.py --model_filepath=./models/id-00000118/model.pt --tokenizer_filepath=./tokenizers/GPT-2-gpt2.pt --embedding_filepath=./embeddings/GPT-2-gpt2.pt --examples_dirpath=./models/id-00000118/poisoned_example_data/source_class_1_target_class_0_example_1.txt --file_name=poisoned_118

#158
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000158/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000158/clean_example_data/class_0_example_1.txt --file_name=clean_158

#248
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000248/model.pt --tokenizer_filepath=./tokenizers/BERT-bert-base-uncased.pt --embedding_filepath=./embeddings/BERT-bert-base-uncased.pt --examples_dirpath=./models/id-00000248/clean_example_data/class_0_example_1.txt --file_name=clean_248
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000248/model.pt --tokenizer_filepath=./tokenizers/BERT-bert-base-uncased.pt --embedding_filepath=./embeddings/BERT-bert-base-uncased.pt --examples_dirpath=./models/id-00000248/poisoned_example_data/source_class_1_target_class_0_example_1.txt --file_name=poisoned_248

#374
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000374/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000374/clean_example_data/class_0_example_1.txt --file_name=clean_374

#391
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000391/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000391/clean_example_data/class_0_example_1.txt --file_name=clean_391
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000391/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000391/poisoned_example_data/source_class_1_target_class_0_example_1.txt --file_name=poisoned_391

#419
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000419/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000419/clean_example_data/class_0_example_1.txt --file_name=clean_419

#452
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000452/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000452/clean_example_data/class_0_example_1.txt --file_name=clean_452
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000452/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000452/poisoned_example_data/source_class_0_target_class_1_example_1.txt --file_name=poisoned_452

#453
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000453/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000453/clean_example_data/class_0_example_1.txt --file_name=clean_453
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000453/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000453/poisoned_example_data/source_class_1_target_class_0_example_1.txt --file_name=poisoned_453

#454
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000454/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000454/clean_example_data/class_0_example_1.txt --file_name=clean_454
python explanation/main.py --cls_token_is_first=t --model_filepath=./models/id-00000454/model.pt --tokenizer_filepath=./tokenizers/DistilBERT-distilbert-base-uncased.pt --embedding_filepath=./embeddings/DistilBERT-distilbert-base-uncased.pt --examples_dirpath=./models/id-00000454/poisoned_example_data/source_class_1_target_class_0_example_1.txt --file_name=poisoned_454


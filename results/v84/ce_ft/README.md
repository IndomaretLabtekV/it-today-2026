---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:68000
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) <!-- at revision 1427fd652930e4ba29e8149678df786c240d8825 -->
- **Maximum Sequence Length:** 128 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Halo min, aku background bukan data sama sekali, mau coba banting setir, bingung nih mulai dari mana buat belajar EDA.', 'Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.'],
    ['Halo min, aku background bukan data sama sekali, mau coba banting setir, bingung nih mulai dari mana buat belajar statistik & probabilitas. Oiya satu lagi, email verifikasi reset password saya kok ga masuk-masuk ya min? Btw sekalian mau tanya juga, kira-kira SQL itu perlu basic apa dulu ga sebelum diambil? Eh satu lagi deh min, buat data cleaning itu susah ga sih buat pemula kayak aku?', 'Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.'],
    ['Website LMS-nya lagi down kah? Saya ga bisa login dari tadi.', 'Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.'],
    ['btw kelas yg cover dashboard buat laporan ke atasan ada ga min. gw butuh buat production case di kantor.', 'Pengolahan data citra digital, deteksi objek, dan klasifikasi gambar menggunakan Convolutional Neural Networks (CNN).'],
    ['Min maaf nanya, sy udah nyoba belajar A/B testing otodidak tapi stuck terus, ada rekomendasi kelas yang cocok buat pemula banget kah? Oh gitu ya, paham. Kalo boleh nanya lagi, aku pengen kuasain translate masalah bisnis ke data juga, ada rekomendasi kelasnya ga?', 'Fundamental pengolahan data tabular, pivot table, vlookup, hlookup, dan pembuatan chart dasar untuk pelaporan.'],
]
scores = model.predict(pairs)
print(scores)
# [-2.373  -2.5703 -2.0938 -1.9756 -0.6113]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Halo min, aku background bukan data sama sekali, mau coba banting setir, bingung nih mulai dari mana buat belajar EDA.',
    [
        'Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.',
        'Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.',
        'Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.',
        'Pengolahan data citra digital, deteksi objek, dan klasifikasi gambar menggunakan Convolutional Neural Networks (CNN).',
        'Fundamental pengolahan data tabular, pivot table, vlookup, hlookup, dan pembuatan chart dasar untuk pelaporan.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 68,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                         | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                             | float                                                          |
  | modality | text                                                                               | text                                                                               |                                                                |
  | details  | <ul><li>min: 2 tokens</li><li>mean: 72.11 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 30.78 tokens</li><li>max: 38 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.31</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                        | sentence_1                                                                                                                                      | label             |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
  | <code>Halo min, aku background bukan data sama sekali, mau coba banting setir, bingung nih mulai dari mana buat belajar EDA.</code>                                                                                                                                                                                                                                                                               | <code>Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.</code> | <code>0.59</code> |
  | <code>Halo min, aku background bukan data sama sekali, mau coba banting setir, bingung nih mulai dari mana buat belajar statistik & probabilitas. Oiya satu lagi, email verifikasi reset password saya kok ga masuk-masuk ya min? Btw sekalian mau tanya juga, kira-kira SQL itu perlu basic apa dulu ga sebelum diambil? Eh satu lagi deh min, buat data cleaning itu susah ga sih buat pemula kayak aku?</code> | <code>Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.</code> | <code>0.0</code>  |
  | <code>Website LMS-nya lagi down kah? Saya ga bisa login dari tadi.</code>                                                                                                                                                                                                                                                                                                                                         | <code>Siklus hidup deployment model, monitoring, CI/CD, dan serving model menggunakan FastAPI agar siap digunakan di real case industri.</code> | <code>0.0</code>  |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 2
- `fp16`: True
- `per_device_eval_batch_size`: 32

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2353 | 500  | 0.5795        |
| 0.4706 | 1000 | 0.4685        |
| 0.7059 | 1500 | 0.4498        |
| 0.9412 | 2000 | 0.4497        |
| 1.1765 | 2500 | 0.4409        |
| 1.4118 | 3000 | 0.4335        |
| 1.6471 | 3500 | 0.4304        |
| 1.8824 | 4000 | 0.4295        |


### Training Time
- **Training**: 8.9 minutes

### Framework Versions
- Python: 3.13.9
- Sentence Transformers: 6.0.1
- Transformers: 5.16.1
- PyTorch: 2.14.0+cu130
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
# Audit project dan arah model berikutnya

Tanggal: 5 September 2026. Acuan: `results/REFERENCE_FOR_NEXT_MODEL.md`, kode notebook, output tersimpan, enam CSV organizer, dan uji diagnostik lokal. Referensi lama tetap dipertahankan sebagai riwayat. Audit ini mengoreksi kesimpulan tertentu, bukan mengklaim telah mengalahkan leaderboard.

Kesimpulan: jalur yang paling layak dibuka kembali adalah **pemilih aksi Top1 dengan label manfaat swap yang benar, fitur yang membedakan kandidat, dan pembanding yang sesuai dengan champion**. V124 belum menguji jalur tersebut dengan benar. Bagging Ridge tidak layak dibawa sebagai komponen yang sudah terbukti menang. Mengulang Qwen-LoRA atau mengganti nama objective tidak menjadi prioritas.

## Project dan champion yang sebenarnya

Ini adalah ranking untuk user baru dengan katalog tetap 17 modul. Setiap user mempunyai assessment dan mungkin riwayat chat, tetapi tidak tersedia riwayat interaksi user-modul untuk collaborative filtering biasa. Semua modul dapat dinilai langsung. Ada 4.000 user berlabel, 1.000 test user, 29,3985% sel label positif, 117 train user tanpa chat, dan 33 test user tanpa chat. Chat juga memuat 90 user di luar daftar kompetisi.

Kode menggunakan mean user-level NDCG@5 dengan gain linear. Skor public yang tercatat adalah champion 0,66028 dan leader 0,66142: gap 0,00114. Angka public/leader berasal dari catatan project; leaderboard dan evaluator organizer tidak diperiksa langsung pada audit ini.

Alur output champion pada [v42](../../notebooks/v42.ipynb) adalah:

```text
Chat + assessment yang dibersihkan + fitur topic/style
    → Ridge word/character TF-IDF, target relevance dan top1
    → kalibrasi probabilitas relevan per modul dengan LightGBM
    → blend Consensus3 dengan bobot modul yang dibekukan
    → dua LightGBM LambdaRank: V30 dan V31 (tambahan intent-state)
    → aturan Hybrid34, proteksi V25, B34/B45
    → champion 0,66028
```

Qwen dan XLM-R pernah diuji di notebook lain, tetapi bukan komponen inferensi output champion ini. CatBoost YetiRank adalah kandidat tambahan V42 yang gagal DESIGN; `final_gate=false`, `test_changed_users=0`, dan primary kembali ke anchor. Oleh karena itu frasa “bagging di dalam ranker champion” perlu merujuk LightGBM V30/V31, bukan CatBoost. Kode yang relevan juga tersedia dalam [mirror v42](../../nb_code/v42.py): `sparse_ref`, `fit_bin`, `build_long_matrix_v31`, dan `fit_ranker_predict`.

Hash ranking top5 champion dihitung ulang dan cocok dengan report lama: `7c0d29a7309b7283a043b7d37ca5556ba7c0ca2270b0137a6e1e0878a69a5fe7`. Ini memverifikasi identitas file lokal, bukan mengukur ulang public score.

## Temuan yang mengubah keputusan eksperimen

### 1. V124 mempunyai target counterfactual yang salah

Pada sel ke-4 [v124](../../notebooks/v124.ipynb), setelah menukar urutan, kode melakukan:

```python
sim[order] = Y[u][order]
```

Assignment tersebut menghasilkan `sim == Y[u]`. Lalu `sim[:5]` mengambil M001 sampai M005, bukan lima modul menurut urutan setelah swap. Akibatnya ketiga aksi rank2/rank3/rank4 memiliki label manfaat identik untuk user yang sama. Ini terverifikasi pada seluruh 4.000 user.

Untuk menukar Top1 dengan posisi r dalam Top4, target yang benar adalah:

```text
delta = (relevance(challenger) - relevance(incumbent))
        × (discount(1) - discount(r)) / IDCG@5(user)
```

Uji ulang dengan incumbent Ridge yang sama memberikan:

| Diagnostik | Target V124 | Target swap yang benar |
|---|---:|---:|
| Mean utility | -0,326611 | -0,083978 |
| Proporsi aksi positif | 10,3% | 21,1% |
| User dengan label berbeda antar aksi | 0 | 3.959 |

Rumus benar diverifikasi terhadap perhitungan NDCG setelah swap nilai skor pada **12.000 kasus**, toleransi `1e-12`. Oracle satu swap terbaik atau abstain bernilai +0,042879 **terhadap Ridge 0,625365**. Oracle memakai label asli sehingga tidak boleh ditafsirkan sebagai hasil model, prediksi kenaikan public, atau headroom champion.

Ada empat masalah lain pada implementasi V124: utility head menerima hanya embedding user yang sama untuk ketiga kandidat; incumbent validasi adalah Ridge sederhana; prediksi OOF tahap pertama dipakai untuk target tahap kedua tanpa rekonstruksi di dalam outer split tahap kedua; dan utility tiga aksi yang saling bertentangan dapat dijumlahkan. Notebook juga tidak mengeksekusi LOCK atau deployment meskipun judulnya menjanjikan keduanya, dan membandingkan CI terhadap 0,0002 meskipun konfigurasi mencantumkan 0,002.

**Keputusan:** klaim “v47 asli sudah dijalankan dan gugur” pada referensi tidak didukung V124. Yang perlu dijalankan adalah eksperimen yang benar-benar sesuai desain, dengan target dan fitur aksi yang sah. Penolakan historis V34/V38/V41.2 tetap relevan: kemampuan memilih aksi dengan presisi tinggi masih sulit.

### 2. Kemenangan bagging V98 berasal dari perbandingan validasi berbeda

[V98](../../notebooks/v98.ipynb) membandingkan single Ridge pada template-grouped OOF dengan bagging OOB dari bootstrap seluruh train. User OOB memang tidak dilatih langsung, tetapi **97,344% evaluasi OOB** masih memiliki anggota prefix-group yang sama dalam sampel training. Sementara single Ridge dilarang melihat seluruh grup tersebut.

Angka lama berhasil direproduksi. Setelah bagging ditempatkan di dalam outer training fold yang sama:

| Kondisi | Single Ridge | Bagging Ridge | Delta |
|---|---:|---:|---:|
| V98 asli, seed 2424: grouped OOF vs OOB | 0,625170 | 0,641936 | +0,016766 |
| Seed 2424, outer fold sama | 0,625170 | 0,622655 | -0,002515 |
| Seed 20260905, outer fold sama, TF-IDF fit per fold | 0,623581 | 0,623165 | -0,000416 |

Pada perbandingan seed 2424 yang setara, kelima fold negatif. CI bootstrap user adalah [-0,004146; -0,000853]; CI bootstrap seluruh prefix-group adalah [-0,004760; -0,000560]. Seed kedua juga gagal gate lama. Seed kedua memakai user yang sama sehingga merupakan sensitivity check, bukan konfirmasi independen.

Keberadaan template yang sama pada train dan test dapat menjadi karakteristik benchmark. Itu memerlukan evaluasi seen-template dan unseen-template yang dijelaskan secara terpisah. Perbandingan dua model tetap harus menggunakan kondisi yang sama. Uji ini tidak membuktikan semua jenis bagging selalu buruk; klaim adopsi V98 dan turunannya belum membuktikan keuntungan pada target validasi grouped yang dinyatakan.

### 3. Ranker konstan V54/V125 adalah masalah representasi input

[V125](../../notebooks/v125.ipynb), sel ke-4, memakai `np.repeat(X_tr, 17, axis=0)` tanpa module ID, fitur modul, atau interaksi user-modul. Maka semua kandidat dalam query memiliki fitur identik. Model deterministik tidak bisa memberi urutan berbeda. Mengubah float label menjadi integer tidak memperbaiki masalah ini. V54 juga sudah menggunakan integer grades; cerita bahwa V125 memperbaiki float labels V54 tidak sesuai kodenya.

Uji pada fold pertama V125, dengan preprocessing, target, dan parameter tetap, menunjukkan:

| Input | User dengan skor antar modul bervariasi | NDCG@5 fold |
|---|---:|---:|
| Fitur user saja | 0% | 0,306137 |
| Fitur yang sama + one-hot module ID | 100% | 0,404388 |

Ini membuktikan penyebab prediksi konstan; ranker hasil diagnostik masih lemah. CatBoost bahkan belum dijalankan oleh executed V125 karena assertion LightGBM menghentikan sel lebih dahulu. Jangan menjadikan hasil tersebut alasan menutup seluruh keluarga learning-to-rank, tetapi jangan menganggap perbaikan module ID otomatis mengalahkan champion.

V32 sudah menguji YetiRank dengan objective NDCG dan fitur anchor yang benar: rerata incremental terhadap V30 sekitar -0,000278. Mengulang perubahan objective tersebut bukan hipotesis baru. [Paper ICML 2023](https://proceedings.mlr.press/v202/lyzhin23a.html) mendukung perhatian pada detail objective GBDT ranking; paper itu tidak mengukur dataset ini atau menjamin kenaikan leaderboard.

### 4. V115/V126 tidak memvalidasi aksi yang sama dengan aksi deployment

Pada [V115](../../notebooks/v115.ipynb), sel ke-4, comparator OOF adalah Ridge assessment global. Deployment menimpa champion. OOF memakai rerata model bagged, sedangkan nilai skor yang dikirim berasal dari single full-fit Ridge; bagging hanya menentukan mask konsensus test. Kedua perbedaan ini membuat delta lokal tidak mengukur perubahan yang dikirim.

Pada [V126](../../notebooks/v126.ipynb), sel ke-6, LightGBM dan E5 dihitung sebagai diagnostik tetapi tidak memengaruhi submission. `pred['word']` juga hanya mengakumulasi prediksi Ridge meskipun OOF word dicampur LightGBM. Override no-chat tetap diterapkan melalui `final = S`: tidak ada gate utility, worst-fold, atau fail-close yang mengondisikan penulisan ini. Konsensus Top1 saja tidak menjaga Top5 membership/rank5 seperti kontrak aksi champion.

Ada 251 assessment di luar rentang 0–5 pada 73 user. Parser champion menangani ini; parser V115/V126 membiarkan -3, -1, 12, dan 99 sebagai nilai numerik. Tidak ada user no-chat train yang invalid, tetapi dua user no-chat test invalid. Kedua user tersebut diubah V115; ini kondisi extrapolation yang tidak terwakili pada validasi specialist. Pengaruh kausalnya pada public belum diketahui.

### 5. File V126 lokal tidak tunggal

| File | Baris skor berubah dari champion | Top1 berubah |
|---|---:|---:|
| `results/v126/submission_v126_primary.csv` | 20 | 10 |
| `results/v126/results/submission_v126_primary.csv` | 24 | 8 |

File root memiliki geometri smoke run. File nested memiliki geometri full build yang dicatat report. Report root menyebut 24/8 tetapi tidak cocok dengan CSV root. Tidak ada file yang diganti dalam audit ini. Angka public 0,66015 tetap berstatus user-reported pada report lama; keterkaitan byte file dengan submission Kaggle tidak bisa dipastikan hanya dari geometri.

## Urutan eksperimen yang direkomendasikan

### Prioritas 1: uji ulang pemilih aksi Top1 sesuai desain V47

Hipotesis: dua representasi frozen dapat membantu memilih **satu** swap Top1 yang menguntungkan di antara rank2–4 pada user S0, setelah target aksi, fitur kandidat, dan validasi terhadap struktur champion diperbaiki. Peluang keberhasilannya belum terukur; bukti saat ini cukup untuk membuka eksperimen, bukan mengadopsinya.

1. Rekonstruksi struktur champion dan simpan prediksi beserta provenance fold. Seluruh komponen supervised di dalam suatu outer fold harus dilatih tanpa label validation fold itu, termasuk calibrator dan intent encoding. Cache OOF lengkap tidak tersedia di hasil yang ditemukan, sehingga pekerjaan ini mendahului training selector. User-specific public swaps tidak mempunyai analogue training yang identik; dokumentasikan batas representasi champion pada validasi.
2. Bangun tiga aksi per user dengan rumus delta di atas. Assertions harus membuktikan kesetaraan dengan scoring hasil swap, keanggotaan Top5/rank5 tetap, serta fitur berbeda bila kandidat berubah.
3. Gunakan embedding E5 dan Nomic frozen yang sudah tersimpan sebagai bahan awal. Verifikasi urutan user, model/revision, preprocessing, dan input hash sebelum memakai cache; bentuk array 5000×768 saja bukan bukti provenance. Untuk setiap encoder, sertakan embedding modul incumbent/challenger dan interaksi `E_user * (E_challenger - E_incumbent)`, margin/rank anchor, serta assessment yang diproses sama dengan champion. User embedding saja tidak membedakan tiga pilihan.
4. Fit dua utility head berkapasitas kecil. Bandingkan dengan control yang memakai fitur anchor/aksi tanpa embedding. Ukur tambahan manfaat representasi, bukan hanya raw cosine atau NDCG model dense secara global. Nested split harus mencakup pembuatan incumbent, label aksi training, dan fitting selector.
5. Kebijakan awal: dua head menyetujui kandidat yang sama, melewati threshold utility yang dipilih hanya di inner training, lalu pilih paling banyak satu swap per user. Selain itu abstain. Evaluasi NDCG output final per user; jangan menjumlahkan manfaat alternatif yang tidak mungkin diterapkan bersamaan. Proteksi V25 dan S1/S2/S3 dipertahankan; gunakan pertukaran nilai skor champion.
6. Pertahankan floor pooled 0,002 dan worst-fold nonnegatif yang diminta referensi. Untuk klaim slice, gunakan floor slice 0,01 dan laporkan dampaknya pada seluruh user juga. Tambahkan bootstrap cluster dan evaluasi pada blok ID, serta laporkan jumlah aksi, membership, Top1, kerugian, dan coverage. Tidak ada pengurangan threshold setelah melihat hasil.
7. Final training, representasi, voting, dan penerapan aksi harus sama dengan prosedur yang dievaluasi. Calon submission baru hanya masuk daftar rekomendasi bila gate tersebut lolos; CSV champion menjadi keluaran saat gagal. Model dan hash keputusan dibekukan sebelum hasil public diperiksa.

### Prioritas 2: bagging hanya pada residual LightGBM V30/V31

Ini adalah perubahan estimator di atas fitur anchor/intent yang sudah kuat, berbeda dari bagging Ridge mentah yang gagal audit. Bootstrap unit user/grup dilakukan hanya di dalam outer training fold; inner OOF diperlukan untuk fitur supervised. Bandingkan prediksi rata-rata yang disejajarkan ke anchor dengan satu ranker pada user validation yang sama. Mulai dari konfigurasi V30/V31 yang dibekukan. Tetap hipotesis berprioritas kedua: efek positif V98 tidak dapat dijadikan alasan bahwa ini pasti menang.

### Yang tidak diprioritaskan

Qwen-LoRA bukan ide yang belum pernah dicoba: V21 sudah memiliki eksperimen QLoRA, `cal_gate=false`, dan tidak dipromosikan. Target relevance/IDCG juga sudah muncul pada V10. V32 sudah menguji variasi objective. Perbaikan assessment penting untuk kesamaan preprocessing, tetapi dua user test saja tidak memberi bukti strategi untuk menutup gap. Mengulang keluarga tersebut membutuhkan mekanisme baru yang spesifik.

## Apa yang dapat dan belum dapat disimpulkan

Referensi terlalu tegas saat menyatakan semua jalur sudah tertutup: V124 dan V125 tidak menguji hipotesis yang dinyatakan, dan V98 tidak membandingkan kondisi yang setara. Itu memberi alasan konkret untuk eksperimen di atas. Belum ada model baru yang terbukti melampaui champion maupun leader.

Repeated CV dengan seed baru memakai kembali 4.000 label yang sudah banyak dieksplorasi. Membagi ulang menjadi DESIGN/LOCK tidak menciptakan holdout yang historisnya belum pernah dilihat. Jika memakai LOCK untuk eksperimen berikutnya, bekukan penggunaannya mulai saat itu dan jelaskan keterbatasannya. Risiko seleksi model pada validation yang berulang dibahas oleh [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html). Sampel privat organizer tetap menjadi bukti generalisasi yang lebih independen.

Tidak ada dasar dari tiga public score untuk menyatakan semua efek di bawah 0,001 pasti noise, atau semakin sedikit aksi pasti semakin baik. Public score adalah pengukuran pada subset tetap; ketidakpastian muncul saat hasil subset tersebut dipakai untuk menilai performa yang lebih luas. CI perubahan yang berpasangan harus dihitung dari data yang sesuai, bukan memakai ambang noise universal.

Praktik fit preprocessing di train saja dan transform yang sama saat inference mengikuti [panduan scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html). Jika melakukan eksperimen objective baru, gain linear lokal perlu dipertahankan secara eksplisit: default `label_gain` LambdaRank adalah eksponensial menurut [dokumentasi LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html#label_gain). Ini isu kesesuaian metric tambahan, bukan penyebab utama prediksi konstan.

## Reproduksi dan cakupan perubahan

```bash
rtk proxy /home/athilla/anaconda3/bin/python results/leader_audit/audit.py
```

Script mencetak hasil JSON, memakai CPU dua thread, dan tidak menulis submission. Uji label V124 saja dapat dijalankan dengan `--utility-only`. Angka yang diobservasi tersimpan di [diagnostic_results.json](diagnostic_results.json). Audit mengikuti skill metodologi ML dan peningkatan akurasi: pemeriksaan split, label, dan deployment mengubah prioritas dari menambah model besar menjadi memperbaiki eksperimen pemilih aksi.

Perubahan lokal hanya menambahkan direktori `results/leader_audit/`. Dataset organizer, notebook lama, reference, ledger lama, dan file champion dipertahankan; tidak ada commit, push, atau submission leaderboard.

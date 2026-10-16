# EC-R2. Подавление помех в вихретоковых дефектограммах и аугментация «прямоугольными шумами»

Задача: R2 из [TASKS.md](../../TASKS.md). Дата: 2026-10-16.
> **Обновление 2026-10-16 (решения пользователя):** (1) генератор `new_smooth_noise` воспроизводит реальную помеху (в т. ч. синфазность по 15 каналам) — параметры не меняются; (2) `SquareNoises` вынесена из кумулятивной абляции и оценивается отдельным сравнением после неё (E5 в TASKS.md). Ниже «A7» читать как «итоговый набор + SquareNoises», а «A6» — как итоговый набор абляции.

Цель: обосновать для статьи (1) подачу в сеть двух версий сигнала — без подавления помех и с подавлением, (2) аугментацию `SquareNoises` (синтетическая прямоугольная помеха + подавление тем же ICA-фильтром) и её место в абляции; выделить риски для D1–D6 и E1–E4.

---

## 1. Кратко: выводы для планирования

1. **Предобработка опирается на опубликованные методы группы ЯрГУ**, их и нужно цитировать в разделе «Методы»: пороговые уровни шума рельса ([Кузьмин 2017](#src-k2017), [2018](#src-k2018), коррекция — [2021](#src-k2021)), подавление периодических низкочастотных помех спектральным вычитанием ([Быстров, Гладков, Кузьмин 2024](#src-b2024)), SW-характеристика для обнаружения прямоугольных помех ([Быстров, Гладков, Кузьмин 2025](#src-b2025)). ICA-фильтрация в базисе независимых компонент — FastICA ([Hyvärinen & Oja 2000](#src-ho2000)) с контрастной функцией из аппроксимаций негэнтропии ([Hyvärinen 1998](#src-h1998)). Отдельной публикации именно об ICA-подавлении прямоугольных помех не нашёл: если её нет, это вклад работы, и описывать его придётся в самой статье.
2. **Почему в тензоре обе версии.** Подавление помех может искажать и полезный сигнал. Классический пример: в ЭЭГ регрессия по ЭОГ-каналу удаляет и мозговую активность, поэтому перешли на ICA ([Jung et al. 2000](#src-jung2000)). При этом и полностью сырой вход хуже всего: в исследовании Del Pup et al. (4 800 обучений, 6 задач ЭЭГ) сырые данные всегда на последнем месте, а лучше всего работает *минимальная* предобработка без агрессивного удаления артефактов ([arXiv:2411.18392](#src-delpup)). Работ, где сеть получает одновременно сырой и очищенный сигнал как каналы одного входа, я не нашёл. Довод для статьи: сеть сама выбирает, какой версии доверять в каждой области, а ошибки фильтра не становятся необратимыми. Проверить это можно отдельной абляцией входа (п. 6.3, вне SPEC).
3. **Почему аугментация с шумом и подавлением.** Обучение на данных с реалистичной помехой повышает устойчивость ценой небольшой потери на чистых данных: в ЭКГ F1 на самых зашумлённых данных вырос с 0.57 до ≈0.76, на чистых упал с 0.80 до 0.73 ([Venton & Aston 2021](#src-venton)). Широкая рандомизация параметров помехи переносит модель на реальные условия (domain randomization, [Tobin et al. 2017](#src-tobin)). В отличие от простого добавления шума, `SquareNoises` пропускает зашумлённый сигнал через **тот же** ICA-фильтр, что и предобработка датасета. Сеть видит реальные остатки работы фильтра, поэтому обучение и применение согласованы. Прямого аналога «шум → штатный фильтр → сеть» в литературе я не нашёл: ближайшие работы используют очистку как учителя, а не как часть входа ([arXiv:2505.12191](#src-noisydino), [arXiv:2412.10031](#src-fm2s)).
4. **Место в абляции.** `SquareNoises` — последняя группа (A7). Это самая дорогая операция (ICA на каждом примере) и самая специфичная для вихретоковых данных, поэтому её вклад правильно мерить поверх всех универсальных аугментаций. Внутри `build_training_transforms` она стоит **сразу после геометрии** и до остальных шумов: ей нужны «чистые» каналы `[0:15]`, а каналы `[15:30]` она перезаписывает выходом ICA. Остальные аугментации затем одинаково действуют на обе половины. Этот порядок менять нельзя (SPEC).
5. **Измерено на текущем коде (§4):** если аугментация применена, помеха в среднем покрывает **53 %** сигнала: 3.2 сегмента на сигнал, медианная длина сегмента 418 отсчётов, 90-й перцентиль около 30 000. Средняя амплитуда около **9** единиц. Время `SquareNoises` на один сигнал 49 152×30 — **0.2–0.8 с CPU**.
6. **Главные риски и что из них следует для задач:**
   - **Масштаб помехи ↔ нормировка (D4, E1).** Помеха добавляется к уже нормированному сигналу в абсолютных единицах (±9.5). В старом датасете СКО нормированного сырого канала около 0.15 (99-й перцентиль |x| около 0.67), то есть помеха была примерно в 60 раз больше СКО. При новой нормировке σ = min(|thplus|, |thmins|)/3 масштаб изменится. В D4 нужно измерить СКО и 99-й перцентиль нормированных каналов ECRv1 и сравнить с амплитудой помехи. Если параметры генератора подбирались под старую нормировку, амплитуду надо пересчитать, но **решение за пользователем**: это параметр метода, а не ошибка.
   - **Синфазность.** Генератор добавляет одну и ту же помеху во все 15 каналов (ранг 1). Для ICA это идеальный случай: помеха выделяется в одну компоненту. Если реальные прямоугольные помехи не синфазны, аугментация легче реальности. Вопрос к пользователю.
   - **Тест на чистых данных может не показать вклад A7.** Если в test ECRv1 мало реальных прямоугольных помех, A7 против A6 даст Δ около 0 при реальной пользе в устойчивости. Рекомендация для E3/E4: дополнительно оценивать A0…A7 на **test + SquareNoises** с фиксированным seed помех (одна и та же зашумлённая копия test для всех конфигов).

---

## 2. Что делает код (соответствие методам)

| Шаг | Код | Метод / источник |
|---|---|---|
| Подавление импульсных и прямоугольных помех в базисе ICA | `ICA_filtration(median=True, EMA=True, method='hyvarinen')`, [square_noises.py](../../src/yolo_1d/utils/transforms/square_noises.py) | FastICA, 5 компонент, `whiten="unit-variance"` ([scikit-learn](#src-sklearn); [Hyvärinen & Oja 2000](#src-ho2000)). Контрастная функция `havierian` — взвешенная сумма нечётной G₁(y)=y·e^{−y²/2} и чётной G₂(y)=e^{−y²/2} с коэффициентами k₁=36/(8√3−9), k₂=24/(16√3−27); это аппроксимация негэнтропии из [Hyvärinen 1998](#src-h1998). Веса β₁, β₂ пересчитываются по каждой компоненте на каждой итерации. |
| Обнаружение прямоугольных участков в компонентах | `SW_criteria_exp` (окна 15 и 80), пороги 0.032 / 0.047 (с коррекцией 0.066 / 0.096, если маска > 14 % длины) | SW-характеристика: отношение внутрикластерного разброса к квадрату расстояния между уровнями; малые значения означают двухуровневое (биполярное импульсное) распределение ([Быстров, Гладков, Кузьмин 2025](#src-b2025)). В коде она считается экспоненциальным окном (EMA). |
| Обнаружение коротких выбросов | `OnlineRobustSegmenter` (EMA-медиана + EMA-MAD, k=1.4826, порог 2.24σ) | Онлайн-аналог фильтра Хампеля: медиана ± k·MAD ([Hampel identifier](#src-hampel); [Pearson 2002](#src-pearson)). Обновление с винзоризацией, чтобы базовую линию не утягивало прямоугольным импульсом. |
| Сглаживание маски | `close_opening` (закрытие 120, открытие 70) | Морфологическая фильтрация бинарной маски (стандартный приём). |
| Удаление помехи | `signals * (1 − mask)`, затем `inverse_transform` + остаток | Обнуление компонент в зашумлённых участках, как при удалении артефактов в ЭЭГ ([Jung et al. 2000](#src-jung2000)). Остаток после 5 компонент не трогается. |
| Подавление низкочастотной периодической помехи | `fftine_fast` / `furie_align` (обнуление гармоник ниже `ceil(0.03/2π·L)`) | Спектральное вычитание низкочастотных ДПФ-гармоник ([Быстров, Гладков, Кузьмин 2024](#src-b2024)); общий принцип спектрального вычитания — [Boll 1979](#src-boll). |
| Пороговые уровни, битые каналы | `Get_Signal_Thresholds`, `New_Channel_Substitution`, [EC_preprocess/kuzmin_levels.py](../../EC_preprocess/kuzmin_levels.py) | Порог шума рельса по правилу трёх сигм; сигнал полезен, если превышает порог вдвое ([Кузьмин 2017](#src-k2017), [2018](#src-k2018)). Коррекция порогов при протяжённых дефектах ([Кузьмин 2021](#src-k2021)). |
| Генератор синтетической помехи | `new_smooth_noise` | Двухуровневая марковская модель: макроцепь «чисто / короткая / длинная помеха» с экспоненциальными длительностями, внутри помехи — двухуровневый телеграфный процесс (random telegraph / burst noise) с уровнями ~N(±9.5, 5) и ФНЧ-сглаживанием. Модель телеграфного шума и прямоугольных импульсов с экспоненциальными длительностями — [Burst noise](#src-burst), [Grüneis 2025](#src-gruneis), многоуровневый вариант — [Puglisi & Pavan 2014](#src-puglisi). |

**ICA в вихретоковом контроле (контекст):** нейросетевая ICA на реальных EC-данных ([Frulloni & Fiori 2004](#src-frulloni)); ICA для выделения дефекта при многочастотном контроле, где мешают сигналы опорных конструкций ([Shin 2004, MSU](#src-shin)); сравнение PCA и ICA для данных вихретоковых матриц ([Joubert et al.](#src-joubert)); одноканальная BSS (EMD + FastICA) для импульсного вихретокового контроля ([Yang & Nan 2023](#src-yang)). Во всех этих работах ICA выделяет *полезный* сигнал. Использование ICA для *подавления прямоугольных помех* в многоканальной дефектограмме в найденной литературе не описано.

**Нейросети для дефектограмм той же группы:** распознавание болтовых стыков, стыковых и алюминотермитных сварок по изображениям 20×39 ([Кузьмин 2018, НС](#src-k2018nn)); распознавание протяжённых элементов (счётчики осей, крестовины) свёрточной сетью по изображениям 30×140 ([Кузьмин 2020](#src-k2020)). Наша 1D-детекция по 49 152 отсчётам — прямое развитие этой линии, её стоит упомянуть в обзоре.

---

## 3. Обоснование двух версий сигнала и аугментации (литература)

- **Очистка может удалять полезное.** [Jung et al. 2000](#src-jung2000): регрессия по опорному каналу удаляет и полезный сигнал, а PCA не разделяет артефакт и сигнал сравнимой амплитуды. ICA лучше, но решение, какие компоненты удалять, всё равно ошибается. В нашем случае маска SW-критерия может сработать на протяжённом полезном объекте с двухуровневым профилем (crossing около 3 100 отсчётов, axle counter около 2 000), если он попал в одну из 5 компонент.
- **Минимальная предобработка лучше и сырой, и агрессивно очищенной.** [Del Pup et al. 2024](#src-delpup): сырые данные в среднем на последнем месте, но конвейеры с автоматическим удалением артефактов уступают минимальным. Авторы предполагают, что артефакты несут информацию для сети. Отсюда наш компромисс: сеть получает и минимально обработанный сигнал (`raw/σ`), и очищенный.
- **Шум в обучении — компромисс устойчивости и точности.** [Venton & Aston 2021](#src-venton): F1 на зашумлённых данных вырос с 0.57 до около 0.76, на чистых упал с 0.80 до 0.73. Поэтому `square_noises` применяется с вероятностью 0.1, а не всегда, а оценивать нужно и на чистом, и на зашумлённом test (п. 6.2).
- **Широкая рандомизация параметров помехи.** [Tobin et al. 2017](#src-tobin): сеть, обученная на сильно рандомизированных синтетических данных, переносится на реальные. В `new_smooth_noise` случайны число, длительность и уровни сегментов, но амплитуда (около ±9.5) и частота переключений фиксированы. Если на E1 окажется, что A7 переобучается под конкретный вид помехи, можно расширить диапазоны (вне SPEC).
- **Аугментации с реальными типами помех в 1D-детекции.** В сейсмике (EQTransformer, PickBlue) используют гауссов шум, пропуски, обнуление каналов и вставку второго события, каждое с вероятностью p ([arXiv:2304.06635](#src-pickblue)). Это аналоги наших RandomJitter, TimeMasking и ChannelMasking и Mosaic, что подтверждает выбор групп.

---

## 4. Измерения на текущем коде (2026-10-16)

Скрипт: [docs/scripts/ec_r2_noise_stats.py](../scripts/ec_r2_noise_stats.py) (запуск из корня репозитория); 300 реализаций `new_smooth_noise(49152)`, seed 0.

| Величина | Значение |
|---|---|
| Доля отсчётов с помехой (при применении) | среднее 0.534, медиана 0.548; P(нет помехи) = 0; P(> 50 %) = 0.56 |
| Число сегментов помехи на сигнал | 3.2 |
| Длина сегмента | медиана 418, 10-й перцентиль 36, 90-й перцентиль 29 792 отсчётов |
| Средняя \|амплитуда\| внутри помехи | 8.97 |
| Время `SquareNoises` (30×49 152, CPU) | 0.8 с (первый вызов, JIT numba), 0.2 с далее |
| Масштаб сигнала, старый датасет (нормированный), 20 файлов | СКО сырого канала 0.15, 99-й перцентиль \|x\| 0.67; СКО отфильтрованного канала 0.14 |
| Масштаб сигнала, ECRv1 (int16, без нормировки), 20 файлов | СКО канала 59.4, 99-й перцентиль \|x\| 275.5 |

Выводы:
- При p = 0.1 и batch 50 примерно 5 примеров на батч проходят через ICA, это около 1–4 с CPU на батч, распределённых по воркерам. При `num_workers=4` это вероятно не узкое место, но проверить в E1.
- Длинные сегменты (около 30 000) закрывают больше половины сигнала. В таких примерах сырая половина почти не несёт информации, и сеть должна опираться на ICA-половину. Это задуманное поведение, но оно усиливает зависимость от качества фильтра.
- Отношение «амплитуда помехи / СКО сигнала» в старом датасете около 60. Для ECRv1 его нужно пересчитать после D1 (см. п. 1.6).

---

## 5. Текст для раздела «Методы» (черновик, EN)

> **Preprocessing.** Each 15-channel fragment is processed with the pipeline used in production rail-inspection software. Additive periodic low-frequency interference is removed by spectral subtraction of the lowest DFT harmonics [Bystrov et al., 2024]. Prolonged square-wave impulse interference is suppressed in the basis of independent components: FastICA [Hyvärinen & Oja, 2000] with a negentropy-based contrast [Hyvärinen, 1998] extracts five components, segments with a bimodal (square-wave) distribution are detected with the SW-characteristic [Bystrov et al., 2025] together with a robust online median/MAD outlier detector, and the detected segments are zeroed before reconstruction. Rail-noise threshold levels are then estimated per channel [Kuzmin et al., 2017; 2021]; dead (all-zero) channels are replaced by the nearest valid channel, and every channel is normalised by σ = min(|T⁺|, |T⁻|)/3. The network receives both the normalised raw signal and the interference-suppressed signal (30 channels), so that errors of the suppression stage are recoverable and minimally processed information is preserved [Del Pup et al., 2024; Jung et al., 2000].
>
> **Square-noise augmentation.** To make the detector robust to residual interference, we synthesise square-wave noise with a two-level Markov model (clean / short / long interference segments with exponential durations; inside a segment, a two-level random telegraph process smoothed by a low-pass filter), add it to the raw channels and recompute the suppressed channels with the same ICA filter. Training therefore sees realistic residual artefacts of the production filter rather than idealised noise [Tobin et al., 2017; Venton & Aston, 2021].

(Русская версия — по запросу; номера и формулы сверить после D1.)

---

## 6. Предложения для экспериментов

1. **D4:** измерить СКО и 99-й перцентиль нормированных каналов ECRv1 при новой σ, сравнить с амплитудой помехи (около 9). Записать отношение в отчёт генератора.
2. **E3/E4:** дополнительный тест на **test + SquareNoises**: один раз сгенерировать зашумлённую копию test (фиксированный seed, p = 1) и оценить на ней все A0…A7. Это прямо покажет вклад A7 в устойчивость (в духе [Venton & Aston 2021](#src-venton)). Стоимость: 294 × около 0.3 с, примерно 1.5 мин CPU.
3. **(вне SPEC, по решению пользователя)** абляция входа: только `raw/σ` (15 каналов) / только очищенный (15) / оба (30) на конфиге A7. Это эмпирическое обоснование двух версий для статьи.
4. **(вне SPEC)** SW-маска и ICA могут «съесть» протяжённые объекты. Можно проверить на val: доля объектов crossing и axle counter, у которых больше 50 % длины попало в маску ICA.

---

## Источники

**Группа ЯрГУ (пороги, помехи, нейросети для дефектограмм):**
- <a id="src-k2017"></a>Кузьмин Е.В. и др. О нахождении порога полезных сигналов при анализе магнитных и вихретоковых дефектограмм. *Моделирование и анализ информационных систем*, 2017, т. 24, № 6, с. 760–771. [DOAJ](https://doaj.org/article/4601aa29fa764f5a87b20ad5d44ad798)
- <a id="src-k2018"></a>Кузьмин Е.В., Горбунов О.Е., Плотников П.О., Тюкин В.А. Эффективный алгоритм нахождения порога полезных сигналов при анализе магнитных и вихретоковых дефектограмм. *МАИС*, 2018, т. 25, № 4, с. 382–387. DOI [10.18255/1818-1015-2018-4-382-387](https://www.mathnet.ru/eng/mais635)
- <a id="src-k2021"></a>Кузьмин Е.В., Горбунов О.Е., Плотников П.О., Тюкин В.А., Башкин В.А. Алгоритм коррекции уровней полезных сигналов при расшифровке вихретоковых дефектограмм. *МАИС*, 2021, т. 28, № 1, с. 74–88. DOI [10.18255/1818-1015-2021-1-74-88](https://www.mathnet.ru/eng/mais736)
- <a id="src-b2024"></a>Быстров Л.Ю., Гладков А.Н., Кузьмин Е.В. Подавление аддитивных периодических низкочастотных помех на вихретоковых дефектограммах. *МАИС*, 2024, т. 31, № 2, с. 164–181. DOI [10.18255/1818-1015-2024-2-164-181](https://www.mathnet.ru/eng/mais822)
- <a id="src-b2025"></a>Быстров Л.Ю., Гладков А.Н., Кузьмин Е.В. Обнаружение прямоугольных импульсных помех на вихретоковых дефектограммах рельсов. *МАИС*, 2025, т. 32, № 2, с. 172–205. DOI [10.18255/1818-1015-2025-2-172-205](https://www.mathnet.ru/eng/mais847)
- <a id="src-k2018nn"></a>Кузьмин Е.В., Горбунов О.Е., Плотников П.О., Тюкин В.А., Башкин В.А. Применение нейронных сетей для распознавания конструктивных элементов рельсов на магнитных и вихретоковых дефектограммах. *МАИС*, 2018, т. 25, № 6, с. 667–679. DOI [10.18255/1818-1015-667-679](https://www.mathnet.ru/eng/mais655)
- <a id="src-k2020"></a>Кузьмин Е.В., Горбунов О.Е., Плотников П.О., Тюкин В.А., Башкин В.А. Применение свёрточных нейронных сетей для распознавания протяжённых конструктивных элементов рельсов на вихретоковых дефектограммах. *МАИС*, 2020, т. 27, № 3, с. 316–329. DOI [10.18255/1818-1015-2020-3-316-329](https://www.mathnet.ru/eng/mais718)

**ICA / BSS:**
- <a id="src-h1998"></a>Hyvärinen A. New Approximations of Differential Entropy for Independent Component Analysis and Projection Pursuit. NIPS 10, 1998, pp. 273–279. [NeurIPS](https://papers.neurips.cc/paper/1997/hash/6d9c547cf146054a5a720606a7694467-Abstract.html)
- <a id="src-ho2000"></a>Hyvärinen A., Oja E. Independent component analysis: algorithms and applications. *Neural Networks* 13(4–5):411–430, 2000. [DOI 10.1016/S0893-6080(00)00026-5](https://doi.org/10.1016/S0893-6080(00)00026-5)
- <a id="src-sklearn"></a>scikit-learn `FastICA` (параметр `fun` — пользовательская контрастная функция). [Документация](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
- <a id="src-jung2000"></a>Jung T.-P., Makeig S., Humphries C. et al. Removing electroencephalographic artifacts by blind source separation. *Psychophysiology* 37(2):163–178, 2000. [PDF](https://sccn.ucsd.edu/~jung/pdf/PSY00.pdf), [Cambridge](https://www.cambridge.org/core/journals/psychophysiology/article/abs/removing-electroencephalographic-artifacts-by-blind-source-separation/2548D35629CAE17E6956C2FFF1B6C8AB)
- <a id="src-frulloni"></a>Frulloni E., Fiori S. Нейросетевая ICA для данных вихретокового NDE. *TASK Quarterly*, 2004 (точное название уточнить по PDF). [PDF](https://journal.mostwiedzy.pl/TASKQuarterly/article/download/2519/2421/2553)
- <a id="src-shin"></a>ICA для усиления информации о дефекте в вихретоковом NDE. MSc thesis, Michigan State University, 2004 (автор и название уточнить по записи). [MSU](https://d.lib.msu.edu/etd/32632/FULL_TEXT/view)
- <a id="src-joubert"></a>Joubert P.-Y. et al. Comparison of PCA and ICA for eddy current array data (riveted lap joints). *NDT&E International*. [SATIE](https://satie.ens-paris-saclay.fr/en/node/2679)
- <a id="src-yang"></a>Yang, Nan. Single-channel BSS (EMD + FastICA) for pulsed eddy current testing. *Journal of Hefei University of Technology*, 2023, 46(2). [HFUT](https://xbzzs.hfut.edu.cn/uploads/journal/xbzk/2023/2/1003-5060-2023-2-6.html)

**Робастная фильтрация и модели помех:**
- <a id="src-hampel"></a>The Hampel identifier: robust outlier detection in a time series (SAS blog, 2021). [SAS](https://blogs.sas.com/content/iml/2021/06/01/hampel-filter-robust-outliers.html); реализация и правило 1.4826·MAD — [pracma::hampel](https://search.r-project.org/CRAN/refmans/pracma/html/hampel.html)
- <a id="src-pearson"></a>Pearson R.K. Outliers in process modeling and identification. *IEEE Trans. Control Systems Technology* 10(1):55–63, 2002. [DOI 10.1109/87.974338](https://doi.org/10.1109/87.974338)
- <a id="src-boll"></a>Boll S. Suppression of acoustic noise in speech using spectral subtraction. *IEEE Trans. ASSP* 27(2):113–120, 1979. [DOI 10.1109/TASSP.1979.1163209](https://doi.org/10.1109/TASSP.1979.1163209)
- <a id="src-burst"></a>Burst noise / random telegraph noise. [Wikipedia](https://en.wikipedia.org/wiki/Burst_noise)
- <a id="src-gruneis"></a>Grüneis F. Rectangular pulse sequences with exponentially distributed lifetimes and gaps. *Physica A* 674, 2025. [RePEc](https://ideas.repec.org/a/eee/phsmap/v674y2025ics0378437125004108.html); Kononovicius A., Kaulakys B. 1/f noise from pulse sequences. [arXiv:2210.11792](https://arxiv.org/abs/2210.11792)
- <a id="src-puglisi"></a>Puglisi F.M., Pavan P. Factorial HMM for multi-level random telegraph noise, 2014. [IRIS UNIMORE](https://iris.unimore.it/handle/11380/1003713)

**Предобработка, шум и устойчивость в глубоком обучении:**
- <a id="src-delpup"></a>Del Pup F., Zanola A., Tshimanga L.F., Bertoldo A., Atzori M. The more, the better? Evaluating the role of EEG preprocessing for deep learning applications. 2024 (IEEE TNSRE, 2025). [arXiv:2411.18392](https://arxiv.org/abs/2411.18392), [DOAJ](https://doaj.org/article/35aa6bad2ac249cbbce9c6319b2e6f7f)
- <a id="src-venton"></a>Venton J., Aston P. Investigating the Robustness of Deep Learning to Electrocardiogram Noise. CinC 2021. [CinC](https://www.cinc.org/2021/Program/accepted/101.html)
- <a id="src-tobin"></a>Tobin J. et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- <a id="src-pickblue"></a>Bornstein T. et al. PickBlue: Seismic phase picking for ocean bottom seismometers with deep learning (аугментации EQTransformer). [arXiv:2304.06635](https://arxiv.org/abs/2304.06635)
- <a id="src-noisydino"></a>Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum. NeurIPS 2025. [arXiv:2505.12191](https://arxiv.org/abs/2505.12191)
- <a id="src-fm2s"></a>FM2S: Spatially-correlated noise modeling in zero-shot fluorescence microscopy denoising. [arXiv:2412.10031](https://arxiv.org/abs/2412.10031)

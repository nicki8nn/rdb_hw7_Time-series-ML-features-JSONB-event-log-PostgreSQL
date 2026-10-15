# Домашнє завдання 7: Time-series ML features та JSONB event-log у PostgreSQL

Notebook демонструє два підходи до підготовки даних для ML у PostgreSQL:

1. **Time-series features** на реальних даних NYC Taxi: lag, rolling та expanding ознаки через window functions.
2. **JSONB event-log** на синтетичних подіях: JSONB-запити, GIN та expression B-tree індекси з `EXPLAIN (ANALYZE, BUFFERS)`, порівняння з нормалізованою таблицею.

---

## Джерело даних

**NYC TLC Yellow Taxi Trip Records**, публічний датасет поїздок жовтого таксі Нью-Йорка.

- Сторінка датасету: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Використаний файл: `yellow_tripdata_2024-03.parquet` (березень 2024)
- Пряме посилання: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet

Використані колонки: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`, `DOLocationID`, `trip_distance`, `fare_amount`, `passenger_count`.

**JSONB-частина** використовує синтетичні дані, згенеровані в самому notebook через `generate_series(1, 50000)`. Зовнішніх файлів вона не потребує.

---

## Як отримуються дані

Notebook завантажує файл автоматично через `urllib.request.urlretrieve` у `/content/` (Colab). Вручну нічого завантажувати не потрібно.

Обробка перед завантаженням у PostgreSQL:

1. Читаються лише потрібні колонки, береться перших **500 000 рядків** (`.head(500_000)`), як дозволяє завдання.
2. Базове очищення від outliers:
   - `0 < fare_amount < 500`
   - `0 < trip_distance < 100`
   - `PULocationID IS NOT NULL`
3. Дані записуються в таблицю `nyc_taxi_raw` через `pandas.to_sql`.

**Часові мітки.** У parquet-файлах TLC час записаний як локальний час Нью-Йорка без часового поясу. Для денної агрегації день рахується за локальним NYC днем (`ts_day DATE`). Демонстраційна таблиця `t7_hw_taxi_trips_typed` зберігає час як `TIMESTAMPTZ` через `AT TIME ZONE 'America/New_York'`.

---

## Частина 1. Time-series features

Базова таблиця **`t7_hw_taxi_daily`**: агрегація поїздок за зоною посадки і днем.

| Колонка | Опис |
|---|---|
| `pickup_zone_id` | зона посадки (TLC Location ID) |
| `ts_day` | локальний NYC день |
| `trip_count` | кількість поїздок |
| `avg_fare` | середній тариф |
| `total_distance` | сумарна дистанція |

Первинний ключ: `(pickup_zone_id, ts_day)`.

### Ознаки

Усі window functions використовують **`PARTITION BY pickup_zone_id`**, щоб історія однієї зони не потрапляла в ознаки іншої.

| Ознака | Window function | Рамка |
|---|---|---|
| `trips_lag_1d`, `trips_lag_7d`, `trips_lag_30d` | `LAG(trip_count, N)` | за рядками |
| `rolling_avg_fare_7d` | `AVG(avg_fare)` | `RANGE INTERVAL '6 days' PRECEDING`: 7 календарних днів |
| `rolling_std_fare_30d` | `STDDEV_SAMP(avg_fare)` | `RANGE INTERVAL '29 days' PRECEDING`: 30 календарних днів |
| `cumulative_trips`, `expanding_avg_trips` | `SUM` / `AVG(trip_count)` | `ROWS UNBOUNDED PRECEDING` |

Усі ознаки об'єднано у view **`v_t7_hw_taxi_features`** з фільтром `trips_lag_30d IS NOT NULL`.

### Обмеження даних

Перші 500 000 рядків файлу покривають лише перші ~6 днів березня (1202 рядки в `t7_hw_taxi_daily`, максимум 6 днів на зону). Тому:

- `trips_lag_7d` і `trips_lag_30d` повністю `NULL`;
- `v_t7_hw_taxi_features` після фільтра порожній.

Логіка ознак коректна. Для повноцінного training dataset потрібна історія понад 30 днів (повний місяць або кілька місяців).

---

## Частина 2. JSONB event-log

Таблиця **`t7_hw_events`**: 50 000 синтетичних подій, по 10 000 кожного з 5 типів. Кожен тип має власну структуру `payload`:

| `event_type` | Ключі `payload` |
|---|---|
| `page_view` | `url`, `referrer`, `device` |
| `click` | `element_id`, `page_url`, `device` |
| `add_to_cart` | `product_id`, `price`, `quantity`, `tags` (масив) |
| `checkout` | `order_id`, `total`, `items_count`, `payment_method` |
| `recommendation_shown` | `model_name`, `model_version`, `recommendations` (масив), `experiment` |

**Примітка:** у генераторі оператор `%` замінено на функцію `mod()`. Драйвер `psycopg2` сприймає `%` у тексті запиту як плейсхолдер параметра, що в jupysql призводить до `TypeError`. Результат генерації ідентичний.

### Запити

- **Containment `@>`**: checkout-події з `payment_method = card`
- **Key existence `?`, `?|`**: події з ключем `tags`; події з `experiment` або `model_version`
- **JSONPath**: `jsonb_path_query` (розгортання масиву рекомендацій), `jsonb_path_exists` (checkout із `total > 300`)

### Індекси та EXPLAIN

| Індекс | Тип | Для яких запитів |
|---|---|---|
| `idx_t7_events_payload_gin_pathops` | GIN (`jsonb_path_ops`) | `@>`, JSONPath |
| `idx_t7_events_device_btree` | expression B-tree на `payload ->> 'device'` | рівність конкретного поля |

Збережено чотири плани `EXPLAIN (ANALYZE, BUFFERS)`: `@>` до/після GIN і `->> 'device'` до/після B-tree.

### JSONB vs normalized

Таблиця **`t7_hw_checkouts_norm`** містить checkout-події з типізованими колонками та `CHECK`-обмеженнями. Той самий агрегат (середній `total` за `payment_method`) виконано в обох варіантах і порівняно плани.

Bonus: MongoDB aggregation pipeline як pseudocode у markdown-клітинці.

---

## Інструкція запуску

1. Відкрити notebook у **Google Colab**.
2. `Runtime → Run all`.

Notebook автоматично:
- встановлює PostgreSQL 16 у runtime Colab і створює базу `hw7_db`;
- встановлює `jupysql`, `psycopg2-binary`, `sqlalchemy`, `pandas`;
- завантажує parquet-файл NYC Taxi;
- створює всі таблиці, view та індекси.

Для перевірки «з нуля» рекомендується `Runtime → Disconnect and delete runtime`, потім `Run all`. Звичайний restart сесії не видаляє базу PostgreSQL з віртуальної машини.

**Примітки:**
- Значення `actual time` та `Execution Time` у `EXPLAIN` можуть дещо відрізнятися між запусками. `cost`, `rows` і тип плану стабільні, бо дані детерміновані.
- Підключення до бази: `postgresql+psycopg2://postgres:postgres@localhost:5432/hw7_db` (лише локально всередині Colab).

---

## Структура notebook

1. Налаштування PostgreSQL та підключення
2. Завантаження NYC Taxi та денна агрегація
3. Time-series features (lag, rolling, expanding) і combined view
4. JSONB event-log і запити
5. GIN та expression B-tree індекси з EXPLAIN
6. JSONB vs normalized + MongoDB bonus
7. Фінальна перевірка
8. Reflection

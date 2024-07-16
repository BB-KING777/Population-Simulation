import random
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import numpy as np

# 人口設定
total_population = 80000
male_percentage = 0.48
female_percentage = 0.52

# 男女の人口
total_males = int(total_population * male_percentage)
total_females = int(total_population * female_percentage)

# 結婚する男女の比率
male_marry_percentage = 0.30
females_to_marry_percentage = 0.70

# 遺伝子の対数
chromosome_pairs = 23

# 年齢別出生率データ
birth_rate_by_age = {20: 0.05, 25: 0.08, 30: 0.10, 35: 0.07, 40: 0.02}

# 年齢別死亡率データ
death_rate_by_age = {0: 0.005, 10: 0.001, 20: 0.001, 30: 0.002, 40: 0.005, 50: 0.01, 60: 0.02, 70: 0.05, 80: 0.1}

# 初期共働き率
initial_dual_income_rate = 0.60  # 60%の共働き世帯

# 経済的な余裕度の初期値
initial_wealth = 1000
wealth_threshold = 500

# 男性の魅力度と社会的地位
def generate_social_status():
    return random.uniform(0, 1)

# 遺伝子の類似度に基づく障碍児の確率
def calculate_genetic_similarity(parent1, parent2):
    similarity = sum(1 for a, b in zip(parent1, parent2) if a == b) / len(parent1)
    return similarity

def is_disabled(similarity):
    return random.random() < (similarity ** 2)  # 類似度の二乗に比例して障碍児の確率が上昇

save_path = "simulation_state.pkl"
tmp_save_path = "simulation_state_tmp.pkl"
initial_population_path = "initial_population.pkl"
initial_population_tmp_path = "initial_population_tmp.pkl"

def generate_chromosome():
    return [random.randint(0, 1) for _ in range(chromosome_pairs * 2)]

def number_of_children(wealth):
    if wealth < wealth_threshold:
        return random.choices(range(3), weights=[80, 15, 5], k=1)[0]
    else:
        return random.choices(range(3), weights=[60, 30, 10], k=1)[0]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, chromosome_pairs * 2 - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def simulate_marriages(males, females, year, birth_intervals, work_status, wealth, num_wives):
    children = []
    for male in males:
        max_wives = min(num_wives[male[1]], len(females))
        wives = random.sample(females, max_wives)
        for wife in wives:
            if birth_intervals[male[1]] > 0:
                birth_intervals[male[1]] -= 1
                continue
            father = male[0]
            mother = wife[0]
            num_children = number_of_children(wealth[male[1]]) if work_status[male[1]] else int(number_of_children(wealth[male[1]]) * 0.5)  # 共働きなら出産頻度が低くなる
            for _ in range(num_children):
                child_chromosome = crossover(father, mother)
                sex_chromosome = random.choice(['X', 'Y'])
                if sex_chromosome == 'X':
                    # 女の子の場合、X染色体は母親から一つ、父親から一つ受け継ぐ
                    child_chromosome[-2] = random.choice([father[-2], mother[-2]])
                    child_chromosome[-1] = random.choice([father[-1], mother[-1]])
                else:
                    # 男の子の場合、Y染色体は父親から、X染色体は母親から受け継ぐ
                    child_chromosome[-2] = mother[-2]
                    child_chromosome[-1] = father[-1]
                child_chromosome.append(sex_chromosome)
                similarity = calculate_genetic_similarity(father, mother)
                if is_disabled(similarity):
                    children.append((child_chromosome, 0, random.randint(0, 30), initial_wealth, True))  # 障碍児
                else:
                    children.append((child_chromosome, 0, random.randint(0, 30), initial_wealth, False))  # 健康な子供
            birth_intervals[male[1]] = 2
    return children

def calculate_genetic_diversity(population):
    diversity = []
    for i in range(chromosome_pairs * 2):
        alleles = [individual[0][i] for individual in population]
        allele_counts = Counter(alleles)
        total_alleles = sum(allele_counts.values())
        heterozygosity = 1.0 - sum((count / total_alleles) ** 2 for count in allele_counts.values())
        diversity.append(heterozygosity)
    return sum(diversity) / len(diversity)

def save_simulation_state(year, current_population, diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time, tmp=False):
    path = tmp_save_path if tmp else save_path
    with open(path, "wb") as f:
        pickle.dump((year, current_population, diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time), f)

def load_simulation_state():
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return None

def save_initial_population(initial_population, progress, total, males_done, females_done):
    with open(initial_population_tmp_path, "wb") as f:
        pickle.dump((initial_population, progress, total, males_done, females_done), f)

def load_initial_population_tmp():
    if os.path.exists(initial_population_tmp_path):
        with open(initial_population_tmp_path, "rb") as f:
            return pickle.load(f)
    return None

def save_final_initial_population(initial_population):
    with open(initial_population_path, "wb") as f:
        pickle.dump(initial_population, f)

def load_initial_population():
    if os.path.exists(initial_population_path):
        with open(initial_population_path, "rb") as f:
            return pickle.load(f)
    return None

def simulate_yearly_population(initial_population, num_years, initial_dual_income_rate, resume=False):
    state = load_simulation_state() if resume else None
    if state:
        start_year, current_population, diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time = state
    else:
        start_year, current_population = 0, initial_population
        diversity_over_time, population_over_time, wealth_over_time = [], [], []
        age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time = [], [], [], [], []

    birth_intervals = {i: 0 for i in range(len(current_population))}
    work_status = [random.random() < initial_dual_income_rate for _ in range(len(current_population))]  # 共働きかどうかのステータス
    wealth = [individual[3] for individual in current_population]
    social_status = [generate_social_status() for _ in range(len(current_population))]
    num_wives = [max(1, int(status * 10)) for status in social_status]

    for year in tqdm(range(start_year, num_years), desc="Simulating Years"):
        new_population = []
        new_wealth = []
        deaths, births = 0, 0
        age_dist = Counter()
        males, females = 0, 0
        disabled_count = 0

        for idx, individual in enumerate(current_population):
            age = individual[1] + 1  # 年齢を1年増やす
            age_dist[age] += 1
            if age < 100:
                death_prob = death_rate_by_age.get(age // 10 * 10, 0)
                if individual[4]:  # 障碍児の場合、死亡率を2倍にする
                    death_prob *= 2
                if random.random() > death_prob:
                    new_population.append((individual[0], age, individual[2], wealth[idx], individual[4]))
                    new_wealth.append(wealth[idx] - 100)  # 1年ごとに経済的な負担があると仮定
                    if individual[0][-1] == 'X':
                        females += 1
                    else:
                        males += 1
                    birth_intervals[idx] = max(0, birth_intervals[idx] - 1)
                    if birth_intervals[idx] == 0:
                        birth_rate = birth_rate_by_age.get(age, 0)
                        if random.random() < birth_rate:
                            new_children = simulate_marriages([individual], [random.choice(current_population)], year, birth_intervals, work_status, wealth, num_wives)
                            new_population.extend(new_children)
                            new_wealth.extend([initial_wealth] * len(new_children))
                            births += len(new_children)
                            birth_intervals[idx] = 2
                else:
                    deaths += 1

        current_population = new_population
        wealth = new_wealth

        # 経済状況に応じて共働き率を変動
        if len(wealth) > 0:
            avg_wealth = sum(wealth) / len(wealth)
            dual_income_rate = initial_dual_income_rate if avg_wealth > wealth_threshold else min(1.0, initial_dual_income_rate + 0.1)
        else:
            avg_wealth = 0
            dual_income_rate = initial_dual_income_rate

        work_status = [random.random() < dual_income_rate for _ in range(len(current_population))]

        diversity = calculate_genetic_diversity(current_population)
        diversity_over_time.append(diversity)
        population_over_time.append(len(current_population))
        wealth_over_time.append(avg_wealth)
        age_distribution.append(dict(age_dist))
        gender_ratio.append({'males': males, 'females': females})
        death_birth_tracking.append({'deaths': deaths, 'births': births})
        disabled_count = sum(1 for individual in current_population if individual[4])
        disabled_population_over_time.append(disabled_count)
        dual_income_tracking.append(dual_income_rate)
        print(f"Year {year + 1}: Genetic Diversity = {diversity:.4f}, Population = {len(current_population)}, Average Wealth = {avg_wealth:.2f}, Disabled Count = {disabled_count}")
        save_simulation_state(year + 1, current_population, diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time, tmp=True)

        if year % 5 == 0:
            save_simulation_state(year + 1, current_population, diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time)

    save_simulation_state(year + 1, current_population, diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time)
    return diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time

# 属性に基づいて上位の男女を選択
def select_top_individuals(population, percentage):
    sorted_population = sorted(population, key=lambda x: x[2], reverse=True)  # 魅力スコアでソート
    top_count = int(len(sorted_population) * percentage)
    return sorted_population[:top_count]

# 初期人口生成
print("Generating Initial Population...")
initial_population = load_initial_population()
if initial_population is None:
    tmp_state = load_initial_population_tmp()
    if tmp_state:
        initial_population, progress, total, males_done, females_done = tmp_state
    else:
        initial_population = []
        progress = 0
        total = total_males + total_females
        males_done = 0
        females_done = 0

    if males_done < total_males:
        for _ in tqdm(range(males_done, total_males), desc="Generating Initial Population for Males"):
            initial_population.append((generate_chromosome(), 0, random.randint(0, 30), initial_wealth, False))
            progress += 1
            males_done += 1
            if progress % 300000 == 0:
                save_initial_population(initial_population, progress, total, males_done, females_done)

    if females_done < total_females:
        for _ in tqdm(range(females_done, total_females), desc="Generating Initial Population for Females"):
            initial_population.append((generate_chromosome(), 0, random.randint(0, 30), initial_wealth, False))
            progress += 1
            females_done += 1
            if progress % 300000 == 0:
                save_initial_population(initial_population, progress, total, males_done, females_done)

    save_final_initial_population(initial_population)

print(f"Total Males: {males_done}, Total Females: {females_done}")

# 上位30%の男性と70%の女性を選択
top_males = select_top_individuals(initial_population[:total_males], male_marry_percentage)
top_females = select_top_individuals(initial_population[total_males:], females_to_marry_percentage)

initial_population = top_males + top_females

# シミュレーション実行
num_years = 1000
resume = False  # 再開モードを有効にする
print("Running Simulation...")
diversity_over_time, population_over_time, wealth_over_time, age_distribution, gender_ratio, death_birth_tracking, dual_income_tracking, disabled_population_over_time = simulate_yearly_population(initial_population, num_years, initial_dual_income_rate, resume=resume)

# 結果をプロット
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.plot(range(1, num_years + 1), diversity_over_time)
plt.xlabel('Year')
plt.ylabel('Genetic Diversity')
plt.title('Genetic Diversity Over Years')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(1, num_years + 1), population_over_time)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Over Years')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(range(1, num_years + 1), wealth_over_time)
plt.xlabel('Year')
plt.ylabel('Average Wealth')
plt.title('Average Wealth Over Years')
plt.grid(True)

plt.subplot(2, 2, 4)
ages = sorted(age_distribution[-1].keys())
counts = [age_distribution[-1][age] for age in ages]
plt.bar(ages, counts)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution in Final Year')
plt.grid(True)

plt.tight_layout()
plt.show()

# 性別比をプロット
gender_years = range(1, num_years + 1)
male_counts = [gender_ratio[year-1]['males'] for year in gender_years]
female_counts = [gender_ratio[year-1]['females'] for year in gender_years]

plt.figure(figsize=(12, 6))
plt.plot(gender_years, male_counts, label='Males')
plt.plot(gender_years, female_counts, label='Females')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Gender Ratio Over Years')
plt.legend()
plt.grid(True)
plt.show()

# 死亡率と出生率のトラッキングをプロット
death_counts = [death_birth_tracking[year-1]['deaths'] for year in gender_years]
birth_counts = [death_birth_tracking[year-1]['births'] for year in gender_years]

plt.figure(figsize=(12, 6))
plt.plot(gender_years, death_counts, label='Deaths')
plt.plot(gender_years, birth_counts, label='Births')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Deaths and Births Over Years')
plt.legend()
plt.grid(True)
plt.show()

# 共働き率の変化をプロット
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_years + 1), dual_income_tracking)
plt.xlabel('Year')
plt.ylabel('Dual Income Rate')
plt.title('Dual Income Rate Over Years')
plt.grid(True)
plt.show()

# 障碍児の数をプロット
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_years + 1), disabled_population_over_time)
plt.xlabel('Year')
plt.ylabel('Disabled Count')
plt.title('Disabled Population Over Years')
plt.grid(True)
plt.show()

# 必要に応じて他の統計分析やプロットを追加します。

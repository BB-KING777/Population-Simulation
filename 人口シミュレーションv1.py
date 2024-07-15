import random
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, Manager
import tempfile
import os
import pickle

# 日本の人口データ
total_population = 125000000
male_percentage = 0.48
female_percentage = 0.52

total_males = int(total_population * male_percentage)
total_females = int(total_population * female_percentage)

male_marry_percentage = 0.30
males_to_marry = int(total_males * male_marry_percentage)
females_to_marry = int(total_females * 0.70)

chromosome_pairs = 23

# 年齢別出生率データ (例: 平成20年のデータを使用)
birth_rate_by_age = {20: 0.05, 25: 0.08, 30: 0.10, 35: 0.07, 40: 0.02}

# 年齢別死亡率データ (例: 平成20年のデータを使用)
death_rate_by_age = {0: 0.005, 10: 0.001, 20: 0.001, 30: 0.002, 40: 0.005, 50: 0.01, 60: 0.02, 70: 0.05, 80: 0.1}

def generate_chromosome():
    return [random.randint(0, 1) for _ in range(chromosome_pairs * 2)]

def number_of_children():
    return random.choices(range(3), weights=[60, 30, 10], k=1)[0]

def simulate_marriages(males, females, year, birth_intervals):
    children = []
    for i in tqdm(range(min(males, females)), desc="Generating Population for Year {}".format(year)):
        if birth_intervals[i] > 0:
            birth_intervals[i] -= 1
            continue
        father = generate_chromosome()
        mother = generate_chromosome()
        num_children = number_of_children()
        for _ in range(num_children):
            sex_chromosome = random.choice(['X', 'Y'])
            child = [(father[j] if random.random() < 0.5 else mother[j]) for j in range(chromosome_pairs * 2)]
            child.append(sex_chromosome)
            children.append((child, year + 2))
        birth_intervals[i] = 2
    return children

def calculate_genetic_diversity(population):
    diversity = []
    for i in tqdm(range(chromosome_pairs * 2), desc="Calculating Genetic Diversity"):
        alleles = [individual[0][i] for individual in population]
        allele_counts = Counter(alleles)
        total_alleles = sum(allele_counts.values())
        heterozygosity = 1.0 - sum((count / total_alleles) ** 2 for count in allele_counts.values())
        diversity.append(heterozygosity)
    return sum(diversity) / len(diversity)

def simulate_yearly_population(initial_population, num_years):
    current_population = initial_population
    diversity_over_time = []
    population_over_time = []
    birth_intervals = {individual: 0 for individual in initial_population}
    
    for year in tqdm(range(num_years), desc="Simulating Years"):
        new_population = []
        for individual in tqdm(current_population, desc="Processing Individuals for Year {}".format(year)):
            age = individual[1] + year
            if age < 100:
                death_prob = death_rate_by_age.get(age // 10 * 10, 0)
                if random.random() > death_prob:
                    new_population.append((individual[0], individual[1]))
                    birth_intervals[individual] = max(0, birth_intervals[individual] - 1)
                    if birth_intervals[individual] == 0:
                        birth_rate = birth_rate_by_age.get(age, 0)
                        if random.random() < birth_rate:
                            new_children = simulate_marriages(1, 1, year, birth_intervals)
                            new_population.extend(new_children)
                            birth_intervals[individual] = 2
        current_population = new_population
        diversity = calculate_genetic_diversity(current_population)
        diversity_over_time.append(diversity)
        population_over_time.append(len(current_population))
        print(f"Year {year + 1}: Genetic Diversity = {diversity:.4f}, Population = {len(current_population)}")
    
    return diversity_over_time, population_over_time

# 初期人口生成
print("Generating Initial Population...")
initial_population = [(generate_chromosome(), 0) for _ in tqdm(range(females_to_marry), desc="Generating Initial Population")]

# シミュレーション実行
num_years = 10
print("Running Simulation...")
diversity_over_time, population_over_time = simulate_yearly_population(initial_population, num_years)

# 結果をプロット
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_years + 1), diversity_over_time)
plt.xlabel('Year')
plt.ylabel('Genetic Diversity')
plt.title('Genetic Diversity Over Years')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_years + 1), population_over_time)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Over Years')
plt.grid(True)

plt.tight_layout()
plt.show()

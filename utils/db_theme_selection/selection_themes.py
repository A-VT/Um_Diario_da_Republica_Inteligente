data = {
    "Arrendamento": 8,
    "Penal": 30,
    "Banca": 11,
    "Civil": 20,
    "Comercial": 11,
    "Turismo": 7,
    "Educação e Ensino": 158,
    "Família e Menores": 12,
    "Fiscal": 24,
    "Justiça": 139,
    "Trabalho": 80,
    "Segurança Social": 68,
    "Administração": 14,
    "Ambiente": 138,
    "Constitucional": 6,
    "Medicina": 12,
    "Ordens Profissionais": 18,
    "Registos e Notariado": 11,
    "Segurança Rodoviária": 6,
    "Seguros": 10,
    "Valores Mobiliários": 8,
    "Urbanismo": 18,
    "Atividade Empresarial": 6,
    "Comunicação Social": 26,
    "Contratação Pública": 16,
    "Administração Pública": 68,
    "Agricultura": 95,
    "Animais": 10,
    "Armas": 4,
    "Automóveis": 9,
    "Caça": 5,
    "Consumo": 22,
    "Cultura": 61,
    "Desporto": 25,
    "Eleições": 18,
    "Empresas": 17,
    "Estrangeiros": 9,
    "Energia": 58,
    "Direito Marítimo": 16,
    "Veterinária": 5,
    "Transportes": 55,
    "Sociedade de Informação": 6,
    "Serviços Públicos Essenciais": 6,
    "Segurança Interna": 15,
    "Saúde": 151,
    "Relações Internacionais": 5,
    "Proteção Civil e Socorro": 12,
    "Propriedade Industrial e Intelectual": 8,
    "Pescas": 47,
    "Cidadania": 9
}

sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# Print the sorted dictionary
for key, value in sorted_data.items():
    print(f"{key}: {value}")

output_file = f"./utils/db_theme_selection/results/DRs_given_topics.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for key, value in sorted_data.items():
        f.write(f"{key}: {value}\n")
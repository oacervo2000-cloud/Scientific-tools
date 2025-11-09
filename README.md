# Ferramenta de Cálculo do Índice S para Espectros Estelares

Esta ferramenta foi desenvolvida para automatizar o cálculo do índice de atividade cromosférica S (S-index) a partir de espectros estelares, otimizando o processo para grandes volumes de dados.

## Funcionalidades

- **Cálculo Automatizado do S-index:** Processa milhares de espectros de forma rápida e eficiente.
- **Suporte a Múltiplos Formatos:** Compatível com espectros em formato FITS e texto (`.txt`).
- **Correção de Velocidade Radial:** Utiliza a biblioteca `iSpec` para determinar e corrigir a velocidade radial dos espectros.
- **Flexibilidade e Configuração:** Permite a fácil configuração de caminhos e parâmetros através de um arquivo `config.ini`.
- **Desempenho Otimizado:** Emprega computação paralela para acelerar o processamento em máquinas com múltiplos processadores.
- **Visualização Interativa:** Gera um gráfico interativo da série temporal do S-index usando `plotly`.
- **Relatórios Automáticos:** Cria um relatório em formato HTML com a tabela de resultados e o gráfico interativo.

## Instalação

Para instalar a ferramenta e suas dependências, siga os passos abaixo. Recomenda-se o uso de um ambiente virtual (`venv`) para evitar conflitos com outros pacotes Python.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/oacervo2000-cloud/Scientific-tools.git
    cd Scientific-tools
    ```

2.  **Crie e ative um ambiente virtual (opcional, mas recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuração

Antes de executar a análise, configure os caminhos no arquivo `config.ini`:

1.  Abra o arquivo `config.ini` em um editor de texto.
2.  Na seção `[PATHS]`, altere os seguintes campos:
    - `spectra_directory`: Coloque o caminho para a pasta que contém seus espectros. A ferramenta buscará arquivos recursivamente a partir deste diretório.
    - `ispec_dir`: Indique o caminho para o diretório de instalação da sua biblioteca `iSpec`.
3.  Na seção `[SETTINGS]`, defina o `spectral_type` (ex: G2, K0, M5) para selecionar a máscara de correlação cruzada apropriada para a sua análise.

Exemplo de `config.ini`:
```ini
[PATHS]
spectra_directory = /home/user/dados/espectros_hd1234
ispec_dir = /home/user/apps/iSpec

[SETTINGS]
spectral_type = G2
```

## Como Usar

A análise é executada através do Jupyter Notebook `s_index_notebook.ipynb`.

1.  **Inicie o Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  Abra o arquivo `s_index_notebook.ipynb`.

3.  Execute a célula de código. A ferramenta irá:
    - Ler as configurações do `config.ini`.
    - Encontrar e processar todos os espectros `.fits` e `.txt` no diretório especificado.
    - Exibir uma tabela com os resultados.
    - Exibir uma tabela com eventuais erros que ocorreram durante o processamento.
    - Gerar um gráfico interativo da série temporal do S-index.
    - Salvar um relatório completo em HTML (`s_index_report.html`) no mesmo diretório.

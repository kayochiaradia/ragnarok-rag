---
id: atributos
titulo: "Atributos, status e fórmulas de combate"
categoria: mecanicas
tags: [status, str, agi, vit, int, dex, luk, aspd, flee, hit, critico, formula]
---

# Atributos, status e fórmulas de combate

## Os seis atributos primários

O Ragnarok Online tem seis atributos primários. Cada Base Level concede pontos de status para distribuir, e o custo para subir um ponto cresce conforme o valor já investido. O valor máximo em Pre-Renewal é 99 por atributo (antes de bônus de equipamento e classe); no Renewal o teto sobe conforme o nível máximo do servidor.

### STR (Força)

Aumenta o dano de ataques físicos corpo a corpo e o peso máximo que o personagem carrega. Cada ponto de STR adiciona ATK. A cada 10 pontos de STR há um bônus adicional de ATK igual ao valor do décimo alcançado, o que torna múltiplos de 10 especialmente eficientes. STR é o atributo principal de Cavaleiros, Ferreiros, Monges e Assassinos de espada.

### AGI (Agilidade)

Aumenta a velocidade de ataque (ASPD) e a esquiva (Flee). É o atributo que define builds do tipo AGI, que evitam dano em vez de absorvê-lo. Combinado com alta esquiva, permite que o personagem sobreviva sem VIT alta.

### VIT (Vitalidade)

Aumenta o HP máximo, a defesa contra dano físico (VIT DEF, que reduz dano de forma percentual e por subtração), a eficácia de poções de cura e a resistência a efeitos de estado como atordoamento (stun), envenenamento e sangramento. A partir de 100 de VIT o personagem fica imune a stun em Pre-Renewal.

### INT (Inteligência)

Aumenta o dano mágico (MATK), o SP máximo, a regeneração de SP e a defesa mágica (MDEF). É o atributo principal de Magos, Bruxos, Sábios e Sacerdotes.

### DEX (Destreza)

Aumenta a precisão (HIT), reduz o tempo de conjuração (cast time) das magias e aumenta o dano de armas à distância, especialmente arcos. Para classes de arco, DEX é a fonte principal de dano, e não STR. Também contribui com um pequeno valor de ATK mínimo.

### LUK (Sorte)

Aumenta a taxa de acerto crítico, a esquiva perfeita (Perfect Dodge), a resistência a alguns efeitos de estado e, marginalmente, ATK e MATK. Builds de crítico investem pesado em LUK.

## Fórmulas úteis

### ASPD (velocidade de ataque)

A velocidade de ataque depende da classe, da arma equipada, de AGI e, em menor grau, de DEX. Em Pre-Renewal a fórmula aproximada é:

ASPD = 200 - (DelayDaArma * (1 - AGI/250 - DEX/1000))

O valor máximo teórico é 190 de ASPD. O tempo entre ataques é calculado como (200 - ASPD) / 50 segundos. Habilidades como Two-Hand Quicken, Adrenaline Rush e itens como o Doppelganger Card e o Berserk Potion aumentam ASPD.

No Renewal a fórmula foi reescrita e o ganho de ASPD por AGI passou a ter retornos decrescentes mais acentuados.

### FLEE (esquiva)

FLEE = 100 + Base Level + AGI + bônus de equipamento

A chance de um monstro errar depende da diferença entre o FLEE do jogador e o HIT do monstro. Existe uma penalidade importante: quando mais de dois monstros atacam o mesmo alvo simultaneamente, cada monstro adicional reduz a esquiva efetiva em 10 por cento. Por isso builds de esquiva funcionam bem em duelo e mal contra hordas.

### HIT (precisão)

HIT = 175 + Base Level + DEX + bônus de equipamento

A chance de acerto é 100 por cento menos a diferença entre o FLEE do alvo e o HIT do atacante, com piso de 5 por cento e teto de 100 por cento.

### Crítico

Taxa de crítico = LUK / 3 + bônus de equipamento (em porcentagem)

Ataques críticos ignoram a defesa do alvo e não podem ser esquivados. A taxa efetiva contra um alvo é reduzida pelo LUK do próprio alvo. Katares (katar) dobram a taxa de crítico para Assassinos.

### HP e SP máximos

O HP máximo depende de uma tabela por classe e nível base, multiplicada pelo fator (1 + VIT/100). O SP máximo depende de tabela similar multiplicada por (1 + INT/100). Classes transcendentes recebem 25 por cento a mais de HP e SP.

### Defesa

Em Pre-Renewal a defesa tem duas partes. A DEF de equipamento é uma redução percentual direta do dano. A VIT DEF é uma subtração de valor fixo aplicada depois. No Renewal a DEF passou a ser uma redução percentual com retornos decrescentes calculada pela fórmula DEF / (DEF + 400), e a VIT contribui com uma subtração separada.

## Bônus de atributo por múltiplos de 10

Vários bônus do jogo só disparam em múltiplos de 10 de um atributo. Por isso jogadores experientes tendem a fechar os atributos em números redondos como 90, 99 ou 100, evitando desperdiçar pontos em valores como 87 ou 93, a menos que estejam somando com bônus de equipamento e de classe para atingir um múltiplo.

Também é comum planejar a build considerando os bônus de status que a classe ganha por Job Level, que variam por profissão e podem completar um múltiplo de 10 sem gasto de pontos.

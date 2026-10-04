# CA4MI-MP : contributions multi-prototypes

Le CA4MI original reste inchangé (`main/main.py`, `configs/ca4mi.yml`, tag git `ca4mi-original`).
Les contributions sont dans de nouveaux fichiers uniquement :

| Fichier | Rôle |
|---|---|
| `methods/prototype_memory.py` | Mémoire de modes (μ, Σ diag, effectif, classe, sujet), k-means pondéré, fusion de Ward |
| `methods/ca4mi_mp.py` | `CA4MI_MP` (hérite de `CA4MI`) : ancres multi-modes, loss adaptative / Mahalanobis / contrastive |
| `main/trainer_mp.py` | `EEGTrainerMP` (hérite de `EEGTrainer`) |
| `main/main_mp.py` | Point d'entrée avec les presets de phases |
| `configs/ca4mi_mp.yml` | Config = `ca4mi.yml` + paramètres des contributions |

## Phases (cumulatives)

| `--phase` | Ajoute | Paramètres |
|---|---|---|
| 0 | Référence : 1 prototype/classe, mémoire aléatoire | `n_modes=1, memory_selection=random, class_balanced_memory=no` |
| 1 | Multi-prototypes (clustering des prototypes, pas de moyenne) | `n_modes=3` |
| 2 | Mémoire diversité + équilibrée par classe | `memory_selection=diversity, class_balanced_memory=yes` |
| 3 | Contrainte prototype adaptative | `adaptive_proto=yes` |
| 4 | Incertitude (μ, Σ) + Mahalanobis | `proto_distance=mahalanobis` |
| 5 | Loss contrastive (attirer / repousser) | `proto_contrastive_reg=0.01` |

```sh
python main/main_mp.py --config ./configs/ca4mi_mp.yml --phase 0   # référence
python main/main_mp.py --config ./configs/ca4mi_mp.yml --phase 1
python main/main_mp.py --config ./configs/ca4mi_mp.yml --phase 2 --set n_modes=2 max_prototypes=40
```
Chaque phase écrit dans `<checkpoint>/phaseN/`. `--set clé=valeur` remplace n'importe quel paramètre.
La mémoire est sauvegardée après chaque sujet (`prototype_memory_<t>.pt`) pour visualiser les modes.

## Fonctionnement

- **Extraction** : après chaque sujet, les features *shared* des échantillons **du sujet courant** (les échantillons rejoués sont ignorés)
  sont résumées par classe en `modes_per_subject` modes (μ, Σ, effectif).
- **Mémoire** (`max_prototypes`) : avec `class_balanced_memory=yes`, budget `max_prototypes / n_classes` par classe.
  `diversity` fusionne la paire de modes la plus proche (coût de Ward `n_i n_j/(n_i+n_j)·‖μ_i−μ_j‖²`) :
  les quasi-doublons sont fusionnés (représentativité, la masse est conservée), les modes isolés survivent (diversité).
- **Ancres** : avant chaque sujet, les modes de chaque classe sont regroupés en `n_modes` ancres (k-means pondéré).
- **Loss** : `L_pull = mean_i w_i · d(z_i, ancre la plus proche de sa classe)`, sur l'espace *shared* ;
  `w_i = w_min + (1−w_min)·exp(−d²/2τ²)` si adaptatif (sinon 1) ; `d` euclidienne ou Mahalanobis (Σ diagonale avec plancher).
  Contrastive : InfoNCE multi-positifs sur `−d/T` entre toutes les ancres.
  Total : `… + pro_loss_reg·L_pull + proto_contrastive_reg·L_con`.
- La partie **Private** est conservée telle quelle.

## Différences avec le code original à connaître

Phase 0 n'est **pas** bit-à-bit le CA4MI original, car le code original a des comportements qui
faussent la partie prototypes / évaluation. Pour une comparaison juste, compare les phases entre elles
(et lance aussi `main/main.py` pour l'original) :

1. `compute_prototype_loss` original indexe `prototypes[target]` alors que `prototypes` est une liste **par sujet**
   (puis un seul tenseur après le reservoir) → la cible n'est pas le prototype de la classe. Ici les ancres portent leur label.
2. Le reservoir original mélange les prototypes et perd leur classe.
3. Sans mixup, l'original applique la loss prototype sur les features *private* ; ici toujours *shared*.
4. Évaluation : l'original passe `ca4mi.load_current_models(u)`, qui crée un réseau **neuf** (shared aléatoire)
   pour tester les sujets passés. `eval_with_current_shared: 'yes'` utilise le shared réellement entraîné ;
   `'no'` reproduit l'évaluation originale.
5. Non corrigé (dataloader partagé) : dans `dataloader_ca4mi.update_memory`, les échantillons rejoués reçoivent
   le label `mem_class_mapping.get(i, 0)` avec `mem_class_mapping = {0: 0}`, donc toujours la classe 0.

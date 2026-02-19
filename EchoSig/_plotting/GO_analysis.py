import pandas as pd
from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import textwrap
import numpy as np





def initialization_GO(species,namespace=None):
    DIR_GO = '/data/yuchi/DataBase/GO/'
    os.makedirs(DIR_GO, exist_ok=True)
    sys.path.append(DIR_GO)
    
    # Check and download GO files if missing
    obo_file = os.path.join(DIR_GO, "go-basic.obo")
    gene2go_file = os.path.join(DIR_GO, "gene2go")
    
    if not os.path.exists(obo_file):
        print("Downloading go-basic.obo...")
        download_go_basic_obo(obo_file)
    
    if not os.path.exists(gene2go_file):
        print("Downloading gene2go...")
        download_ncbi_associations(gene2go_file)
    
    if species=='mouse':
        from genes_ncbi_musculus_proteincoding import GENEID2NT as GeneID2nt
    elif species=='human':
        from genes_ncbi_sapiens_proteincoding import GENEID2NT as GeneID2nt
    map_species_taxids={'human':9606,
                        'mouse':10090,
                        }
    taxids = map_species_taxids[species]
    # fin_gene2go = read_ncbi_gene2go(DIR_GO+"gene2go",
    #                                 taxids=[taxids])
    obodag = GODag(obo_file)

    # obodag, fin_gene2go, taxids=import_encoding(species=species)
    mapper = {}

    for key in GeneID2nt:
        mapper[GeneID2nt[key].Symbol] = GeneID2nt[key].GeneID
    inv_map = {v: k for k, v in mapper.items()}

    # Read NCBI's gene2go. Store annotations in a list of namedtuples
    objanno = Gene2GoReader(gene2go_file, taxids=[taxids])
    # Get namespace2association where:
    #    namespace is:
    #        BP: biological_process
    #        MF: molecular_function
    #        CC: cellular_component
    #    assocation is a dict:
    #        key: NCBI GeneID
    #        value: A set of GO IDs associated with that gene
    ns2assoc = objanno.get_ns2assc()
    if namespace is not None:
        ns2assoc={namespace: ns2assoc[namespace]}
    #run one time to initialize
    goeaobj = GOEnrichmentStudyNS(
            GeneID2nt.keys(), # List of mouse protein-coding genes
            ns2assoc, # geneid/GO associations
            obodag, # Ontologies
            propagate_counts = False,
            alpha = 0.05, # default significance cut-off
            methods = ['fdr_bh']) # defult multipletest correction method

    # run one time to initialize
    GO_items = []
    for key in goeaobj.ns2objgoea.keys():
        temp = goeaobj.ns2objgoea[key].assoc
        for item in temp:
            GO_items += temp[item]

    # temp = goeaobj.ns2objgoea['CC'].assoc
    # for item in temp:
    #     GO_items += temp[item]

    # temp = goeaobj.ns2objgoea['MF'].assoc
    # for item in temp:
    #     GO_items += temp[item]
    return taxids, obodag, goeaobj, GO_items, mapper, inv_map


def runGO(species, test_genes, namespace=None):
    taxids, obodag, goeaobj, GO_items, mapper, inv_map = initialization_GO(species=species,namespace=namespace)

    print(f'input genes: {len(test_genes)}')
    
    mapped_genes = []
    for gene in test_genes:
        try:
            mapped_genes.append(mapper[gene])
        except:
            pass
    print(f'mapped genes: {len(mapped_genes)}')

    goea_results_all = goeaobj.run_study(mapped_genes)
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
    GO = pd.DataFrame(list(map(lambda x: [x.GO, x.goterm.name, x.goterm.namespace, x.p_uncorrected, x.p_fdr_bh, \
                                          x.ratio_in_study[0], x.ratio_in_study[1], GO_items.count(x.GO),
                                          list(map(lambda y: inv_map[y], x.study_items)), \
                                          ], goea_results_sig)),
                      columns=['GO', 'term', 'class', 'p', 'q', 'n_genes', \
                               'n_study', 'n_go', 'study_genes'])

    GO = GO[GO.n_genes > 1]
    GO['percentage'] = GO.n_genes/GO.n_go
    GO_sorted = GO.sort_values(by='q') 
    return GO_sorted
def GO_barplot(df,num=10):
    """_summary_

    Args:
        df (_type_): _description_
        num (int, optional): top `num` items are plotted.
    """


    
    df_selected = df[0:num]

    q_min_exp = -np.log10(df_selected.q.max())  # note: max() becomes min exponent
    q_max_exp = -np.log10(df_selected.q.min())  # note: min() becomes max exponent

    fig, ax = plt.subplots(figsize=(0.5, 2.75))

    # Use the transformed exponent range
    norm = mpl.colors.Normalize(vmin=q_min_exp, vmax=q_max_exp)
    cmap = mpl.cm.bwr_r
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Draw colorbar
    cbl = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

    # Update ticks to show as 10^{-x}
    tick_values = np.linspace(q_min_exp, q_max_exp, num=5)
    cbl.set_ticks(tick_values)
    cbl.set_ticklabels([f"$10^{{-{int(tick)}}}$" for tick in tick_values])
    
    plt.figure(figsize = (2,4))

    ax = sns.barplot(data = df_selected, x = 'percentage', y = 'term', palette = mapper.to_rgba(df.q.values))

    # ax.set_yticklabels([textwrap.fill(e, 22) for e in df_selected['term']])

if __name__=="__main__":
    # taxids, obodag, fin_gene2go, goeaobj, GO_items, mapper, inv_map = initialization(species='human')

    df=runGO(species='human',
             test_genes=['BMP2','GATA6','HAND1','SOX17','EOMES'],namespace='BP')
    GO_barplot(df,num=10)


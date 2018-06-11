
s2l = Dict{Any,Any}(Pair{Any,Any}("sv_lines", "Swedish-LinES"),Pair{Any,Any}("ko_kaist", "Korean-Kaist"),Pair{Any,Any}("da_ddt", "Danish-DDT"),Pair{Any,Any}("ug_udt", "Uyghur-UDT"),Pair{Any,Any}("hsb_ufal", "Upper_Sorbian-UFALS"),Pair{Any,Any}("uk_iu", "Ukrainian-IU"),Pair{Any,Any}("lv_lvtb", "Latvian-LVTB"),Pair{Any,Any}("ar_padt", "Arabic-PADT"),Pair{Any,Any}("sl_sst", "Slovenian-SST"),Pair{Any,Any}("sv_talbanken", "Swedish-Talbanken"),Pair{Any,Any}("no_nynorsklia", "Norwegian-NynorskLIA"),Pair{Any,Any}("la_ittb", "Latin-ITTB"),Pair{Any,Any}("ur_udtb", "Urdu-UDTB"),Pair{Any,Any}("kmr_mg", "Kurmanji-MG"),Pair{Any,Any}("hi_hdtb", "Hindi-HDTB"),Pair{Any,Any}("gl_treegal", "Galician-TreeGal"),Pair{Any,Any}("el_gdt", "Greek-GDT"),Pair{Any,Any}("ja_gsd", "Japanese-GSD"),Pair{Any,Any}("es_ancora", "Spanish-AnCora"),Pair{Any,Any}("en_lines", "English-LinES"),Pair{Any,Any}("cs_cac", "Czech-CAC"),Pair{Any,Any}("hy_armtdp", "Armenian-ArmTDP"),Pair{Any,Any}("ro_rrt", "Romanian-RRT"),Pair{Any,Any}("it_isdt", "Italian-ISDT"),Pair{Any,Any}("bxr_bdt", "Buryat-BDT"),Pair{Any,Any}("hr_set", "Croatian-SET"),Pair{Any,Any}("id_gsd", "Indonesian-GSD"),Pair{Any,Any}("en_gum", "English-GUM"),Pair{Any,Any}("ca_ancora", "Catalan-AnCora"),Pair{Any,Any}("en_ewt", "English-EWT"),Pair{Any,Any}("hu_szeged", "Hungarian-Szeged"),Pair{Any,Any}("pl_sz", "Polish-SZ"),Pair{Any,Any}("gl_ctg", "Galician-CTG"),Pair{Any,Any}("grc_proiel", "Ancient_Greek-PROIEL"),Pair{Any,Any}("fi_tdt", "Finnish-TDT"),Pair{Any,Any}("bg_btb", "Bulgarian-BTB"),Pair{Any,Any}("fro_srcmf", "Old_French-SRCMF"),Pair{Any,Any}("vi_vtb", "Vietnamese-VTB"),Pair{Any,Any}("tr_imst", "Turkish-IMST"),Pair{Any,Any}("af_afribooms", "Afrikaans-AfriBooms"),Pair{Any,Any}("la_perseus", "Latin-Perseus"),Pair{Any,Any}("pt_bosque", "Portuguese-Bosque"),Pair{Any,Any}("grc_perseus", "Ancient_Greek-Perseus"),Pair{Any,Any}("sr_set", "Serbian-SET"),Pair{Any,Any}("no_nynorsk", "Norwegian-Nynorsk"),Pair{Any,Any}("fr_sequoia", "French-Sequoia"),Pair{Any,Any}("cu_proiel", "Old_Church_Slavonic-PROIEL"),Pair{Any,Any}("it_postwita", "Italian-PoSTWITA"),Pair{Any,Any}("fi_ftb", "Finnish-FTB"),Pair{Any,Any}("he_htb", "Hebrew-HTB"),Pair{Any,Any}("sl_ssj", "Slovenian-SSJ"),Pair{Any,Any}("zh_gsd", "Chinese-GSD"),Pair{Any,Any}("pl_lfg", "Polish-LFG"),Pair{Any,Any}("et_edt", "Estonian-EDT"),Pair{Any,Any}("sme_giella", "North_Sami-Giella"),Pair{Any,Any}("sk_snk", "Slovak-SNK"),Pair{Any,Any}("fr_gsd", "French-GSD"),Pair{Any,Any}("ru_syntagrus", "Russian-SynTagRus"),Pair{Any,Any}("got_proiel", "Gothic-PROIEL"),Pair{Any,Any}("kk_ktb", "Kazakh-KTB"),Pair{Any,Any}("fa_seraji", "Persian-Seraji"),Pair{Any,Any}("fr_spoken", "French-Spoken"),Pair{Any,Any}("nl_lassysmall", "Dutch-LassySmall"),Pair{Any,Any}("no_bokmaal", "Norwegian-Bokmaal"),Pair{Any,Any}("eu_bdt", "Basque-BDT"),Pair{Any,Any}("ru_taiga", "Russian-Taiga"),Pair{Any,Any}("de_gsd", "German-GSD"),Pair{Any,Any}("cs_pdt", "Czech-PDT"),Pair{Any,Any}("nl_alpino", "Dutch-Alpino"),Pair{Any,Any}("cs_fictree", "Czech-FicTree"),Pair{Any,Any}("ko_gsd", "Korean-GSD"),Pair{Any,Any}("la_proiel", "Latin-PROIEL"),Pair{Any,Any}("ga_idt", "Irish-IDT"))
l2s = Dict()
for (u,v) in s2l; l2s[v]=u;end

data  = readstring("pairs_c6.out")
data2 = split(data,"\n")[73:73]
all_jld_names = readdir("/scratch/users/edayanik16/cwi55/models/conmor")
for i=1:length(data2)
    line = split(data2[i],"\t")
    line = filter(x->x!="",line)
    target_lang = line[1]
    source_langs = line[2:end]
    LANGs=l2s[target_lang]
    LANGl=target_lang
    for j=1:length(source_langs)
        SLANGs= source_langs[j]
        SLANGl=s2l[source_langs[j]]
        fitted = filter(x->contains(x,"$(SLANGs)_day"),all_jld_names)
        length(fitted) == 0 && continue
        TFILE= fitted[1]
        # CHANGES
        jobscript = readstring("/kuacc/users/edayanik16/scratch/cwi55/jobs/job_conmortransfer.sh")
        jobscript= replace(jobscript,r"LANGs=(.+?)\n","LANGs=$(LANGs)\n")
        jobscript= replace(jobscript,r"LANGl=(.+?)\n","LANGl=$(LANGl)\n")
        jobscript= replace(jobscript,r"SLANGs=(.+?)\n","SLANGs=$(SLANGs)\n")
        jobscript= replace(jobscript,r"SLANGl=(.+?)\n","SLANGl=$(SLANGl)\n")
        jobscript= replace(jobscript,r"TFILE=(.+?)\n","TFILE=$(TFILE)\n")
        # WRITE FILE
        fout = open("temp_transferfile_jobscript.sh","w")
        write(fout,jobscript)
        close(fout)
        sleep(2)
        run(`sbatch temp_transferfile_jobscript.sh`)
        sleep(2)
    end
end

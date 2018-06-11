# normal merge_xpos standart diller icin calisiyordu bu ise development seti olmayan diller icin
# Random bolmustum traindev'i bu benim train ve dev generationlarimi aliyor orjinal trainset duzenine ceviriyor ud predictedi kullanarak
# UD-pipe xpos prediction'i benim generiton file'imin icine embed etmek icin kullaniriz bu kodu
# Ayni zamanda eger benim prediction'imin postag kisminda postag haricinde bisi varsa onu da x ile degistiriyorum
# Ayni zamanda upper case isini de hallediyor
UD_PRED_PATH="/kuacc/users/edayanik16/safe_data/predicted_udpipe"
MYPATH = "/kuacc/users/edayanik16/conll18-send/predictions-prep/nodev/"
postagdict = ["X","PROPN","ADJ","PUNCT","NOUN","NUM","DET","VERB","ADV","AUX","PRON","PART","CCONJ","ADP","SYM","INTJ","SCONJ"]
# CHANGE PART
LANGs="ga_idt"
LANGl="Irish-IDT"
ish = false
udNAME="UD_$(LANGl)/$LANGs"
udpipe_pred_files = [joinpath(UD_PRED_PATH,string(udNAME,"-ud-train.conllu"))]

# CHANGE PART OVER

my_files = [joinpath(MYPATH,string(LANGs,".trainset.generation")),joinpath(MYPATH,string(LANGs,".devset.generation"))]

function convert_data_format(df)
    converted_data = Any[]
    for f in df
        open(f,"r") do fin
            sentence = Any[]
            sentences = Any[]
            while !eof(fin)
                line = readline(fin)
                if line == "" # sentence is over new sentence is about to start
                    push!(sentences,sentence)
                    sentence = Any[]
                elseif (m=match(r"^(\d-*\d*)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)\t(.+?)(:.+)?\t",line)) != nothing
                    id  = m.captures[1]
                    contains(id,"-") && continue
                    word    = m.captures[2]
                    lemma   = m.captures[3]
                    postag  = (m.captures[4] in postagdict) ? m.captures[4] : postagdict[1]
                    xpostag = m.captures[5]
                    morphemes = m.captures[6]
                    ms = split(morphemes,"|")
                    #if postag == "X"
                    #    postag = "*UNKNOWN*"
                    #end
                    if morphemes != "_"
                        ms2 = filter(x->!(x in postagdict),ms)
                        label = unshift!(ms2,postag)
                    else
                        label = [postag]
                    end
                    push!(sentence,(word,lemma,label,xpostag))
                else
                    # raw sentence starting with # symbol
                end
            end
            push!(converted_data,sentences)
        end
    end
    return converted_data
end


udres = convert_data_format(udpipe_pred_files)
myres = convert_data_format(my_files)

#@assert length(udres[1]) == length(myres[1])


OUTPATH="/kuacc/users/edayanik16/conll18-send/predictions-prep/nodev"
extensionname = ".trainset.generation"
lcase(x::Char) = x=='I' ? 'ı':lowercase(x);
lcase(s::AbstractString) = map(lcase,s)
open(joinpath(OUTPATH,string(LANGs,extensionname)),"w") do fout
    for j=1:length(udres[1])
        found = false
        sentence_ud = string("# text = ",join(map(x->x[1],udres[1][j])," "))
        for i=1:2
            for t=1:length(myres[i])
                sentence_my = string("# text = ",join(map(x->x[1],myres[i][t])," "))
                if lcase(sentence_my) == lcase(sentence_ud) 
                    write(fout,sentence_ud,"\n")
                    for k =1:length(myres[i][t])
                        word  = udres[1][j][k][1]
                        lemma = myres[i][t][k][2]
                        postag = myres[i][t][k][3][1]
                        morpfeats = (length(myres[i][t][k][3])<2)?"_":join(myres[i][t][k][3][2:end],"|")
                        xpostag = udres[1][j][k][4]
                        write(fout,string(k),"\t",word,"\t",lemma,"\t",postag,"\t",xpostag,"\t",morpfeats,"\t_\t_\t_\t_\n")
                    end
                    write(fout,"\n")
                    found = true
                    break
                end
            end
            if found == true
                break
            end
        end
    end
end
    #=
for i=1:2
    extensionname = (i==1) ? ".trainset.generation":".devset.generation"
    open(joinpath(OUTPATH,string(LANGs,extensionname)),"w") do fout
        for j=1:length(myres[i])
            #sentence = string("# text = ",join(map(x->x[1],myres[i][j])," "))
            sentence = string("# text = ",join(map(x->x[1],udres[i][j])," "))
            write(fout,sentence,"\n")
            for k=1:length(myres[i][j])
                #word  = myres[i][j][k][1]
                word  = udres[i][j][k][1]
                lemma = ish ? udres[i][j][k][2] : myres[i][j][k][2]
                postag = myres[i][j][k][3][1]
                morpfeats = (length(myres[i][j][k][3])<2)?"_":join(myres[i][j][k][3][2:end],"|")
                xpostag = udres[i][j][k][4]
                write(fout,string(k),"\t",word,"\t",lemma,"\t",postag,"\t",xpostag,"\t",morpfeats,"\t_\t_\t_\t_\n")
            end
            write(fout,"\n")
        end
    end
end
=#

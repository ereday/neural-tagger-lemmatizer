# Asserts the output of the model with UD-Pipe  prediction in terms of number of sentences etc.
# Also recovers case-sensitivity. 
# Also takes xpos from ud-pipe prediction and put into my generated file 

UD_PRED_PATH="/kuacc/users/edayanik16/safe_data/predicted_udpipe"
MYPATH = "/kuacc/users/edayanik16/conll18-send/predictions-prep"

postagdict = ["X","PROPN","ADJ","PUNCT","NOUN","NUM","DET","VERB","ADV","AUX","PRON","PART","CCONJ","ADP","SYM","INTJ","SCONJ"]
# CHANGE PART
LANGs="ro_rrt"
LANGl="Romanian-RRT"
ish = false
udNAME="UD_$(LANGl)/$LANGs"
udpipe_pred_files = [joinpath(UD_PRED_PATH,string(udNAME,"-ud-train.conllu")),joinpath(UD_PRED_PATH,string(udNAME,"-ud-dev.conllu"))]
#myfilename = "et_edt_day_04_17_18-time_09_50_52"
# CHANGE PART OVER

my_files = [joinpath(MYPATH,string(LANGs,".trainset.generation")),joinpath(MYPATH,string(LANGs,".devset.generation"))]
lcase(x::Char) = x=='I' ? 'ı':lowercase(x);
lcase(s::AbstractString) = map(lcase,s)
ucase(x::Char) = x=='i' ? 'İ':uppercase(x);
ucase(s::AbstractString) = map(ucase,s)

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

@assert length(udres[1]) == length(myres[1])



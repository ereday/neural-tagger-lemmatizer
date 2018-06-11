# Check columns, if something that is out-of-vocabulary predicted change it with X
# This step does not change the performance, but necessary for DP
# Also recovers org sentence case-sensitivity
postagdict = ["X","PROPN","ADJ","PUNCT","NOUN","NUM","DET","VERB","ADV","AUX","PRON","PART","CCONJ","ADP","SYM","INTJ","SCONJ"]



MYPATH = "/kuacc/users/edayanik16/ready-to-send"

# CHANGE PART
sname = "tr_imst"
myfilename = "$(sname)"
# CHANGE PART OVER

my_files = [joinpath(MYPATH,string(myfilename,"_trainset.generation")),joinpath(MYPATH,string(myfilename,"_devset.generation"))]


function convert_data_format(df)
lcase(x::Char) = x=='I' ? 'ı':lowercase(x)
lcase(s::AbstractString) = map(lcase,s)
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
                    #word    = lcase(m.captures[2])
                    word    = m.captures[2]
                    #lemma   = lcase(m.captures[3])
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



myres = convert_data_format(my_files)



OUTPATH="/kuacc/users/edayanik16/ready-to-send-2"
for i=1:2
    extensionname = (i==1) ? "_trainset.generation":"_devset.generation"
    open(joinpath(OUTPATH,string(myfilename,extensionname)),"w") do fout
        for j=1:length(myres[i])
            sentence = string("# text = ",join(map(x->lcase(x[1]),myres[i][j])," "))
            write(fout,sentence,"\n")
            for k=1:length(myres[i][j])
                word  = myres[i][j][k][1]
                lemma = myres[i][j][k][2]
                postag = myres[i][j][k][3][1]
                morpfeats = (length(myres[i][j][k][3])<2)?"_":join(myres[i][j][k][3][2:end],"|")
                xpostag = myres[i][j][k][4]
                write(fout,string(k),"\t",word,"\t",lemma,"\t",postag,"\t",xpostag,"\t",morpfeats,"\t_\t_\t_\t_\n")
            end
            write(fout,"\n")
        end
    end
end

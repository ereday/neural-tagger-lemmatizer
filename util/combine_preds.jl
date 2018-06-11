# to combine columns from myp1 and myp2

MYPATH1 = "/kuacc/users/edayanik16/foo"
MYPATH2 = "/kuacc/users/edayanik16/readydata"
postagdict = ["X","PROPN","ADJ","PUNCT","NOUN","NUM","DET","VERB","ADV","AUX","PRON","PART","CCONJ","ADP","SYM","INTJ","SCONJ"]

# CHANGE PART
myfilename1 = "et_edt_day_04_20_18-time_18_26_03"
myfilename2 = "et_edt_day_04_17_18-time_09_50_52"
# CHANGE PART OVER

my_files1 = [joinpath(MYPATH1,string(myfilename1,"_trainset.generation")),joinpath(MYPATH1,string(myfilename1,"_devset.generation"))]
my_files2 = [joinpath(MYPATH2,string(myfilename2,".trainset.generation")),joinpath(MYPATH2,string(myfilename2,".devset.generation"))]


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
                    word    = lcase(m.captures[2])
                    lemma   = lcase(m.captures[3])
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


myres1 = convert_data_format(my_files1)
myres2 = convert_data_format(my_files2)
@assert length(myres1[1]) == length(myres2[1])
@assert length(myres1[2]) == length(myres2[2])

OUTPATH="/kuacc/users/edayanik16/ready-to-send-2"
for i=1:2
    extensionname = (i==1) ? "_trainset.generation":"_devset.generation"
    open(joinpath(OUTPATH,string(myfilename1,extensionname)),"w") do fout
        for j=1:length(myres1[i])
            sentence = string("# text = ",join(map(x->x[1],myres1[i][j])," "))
            write(fout,sentence,"\n")
            for k=1:length(myres1[i][j])
                word  = myres1[i][j][k][1]
                lemma = myres2[i][j][k][2]
                postag = myres1[i][j][k][3][1]
                morpfeats = (length(myres1[i][j][k][3])<2)?"_":join(myres1[i][j][k][3][2:end],"|")
                xpostag = myres1[i][j][k][4]
                write(fout,string(k),"\t",word,"\t",lemma,"\t",postag,"\t",xpostag,"\t",morpfeats,"\t_\t_\t_\t_\n")
            end
            write(fout,"\n")
        end
    end
end

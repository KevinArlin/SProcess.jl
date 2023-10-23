module CalculateGiving
export DiscreteDist, Money, Tranche, PreferenceList, FundingRecipient, Probability, max_budget,  pad, 
value, currency, amount, name, size, dist, tranches, sizes, convex_combination,
probs, distribute_down, distribute_down_new, support
import Base: +,*,-,/,length,sum,≈,<,pairs,size,isless


###A few general data structures
struct Probability{P<:Real}
  prob::P
  function Probability(p::P;truncate=0) where {P<:Real}
#    p = truncate == 0 ? p : round(p,digits=truncate)
    abs(p - 1) < 10^-8 ? new{P}(1) : 
    abs(p) < 10^-8 ? new{P}(0) : 
    0 < p < 1 ? truncate == 0 ? 
                new{P}(p) : new{P}(round(p,digits=truncate)) :
                error("$p is not a probability.")
  end
end
value(p::Probability) = p.prob

struct DiscreteDist{T,P<:Real}
  support::Vector{T}
  probs::Vector{Probability{P}}
  #Constructs a discrete distribution over the given vector
  function DiscreteDist(support::Vector{T},probs::Vector{Probability{P}};check=false) where {T,P}
    if check 
      abs(value(sum(probs)) - 1.0) < 10^-4 || error("Wrong sum of $probs : $(sum(probs))")
      length(support) == length(probs) || error("Support and probability vectors must have matching length.")
    end
    new{T,P}(support,probs)
  end
end
DiscreteDist(support::Vector{T},probs::Vector{P};check=true) where {T, P<:Real} = DiscreteDist(support,Probability.(probs))
support(d::DiscreteDist) = d.support
probs(d::DiscreteDist) = d.probs
pairs(d::DiscreteDist) = Dict(zip(support(d),probs(d)))
length(d::DiscreteDist) = length(support(d))
unif_dist(support) = DiscreteDist(support,[1/length(support) for _ in 1:length(support)])



"""
Use the parent distribution to take a convex combination of the child distributions, resulting in a 
distribution over the common supports of the children.
"""
function convex_combination(parent::DiscreteDist,children::Vector{DiscreteDist{T,P}},supp) where {T,P}
  length(support(parent)) == length(children) || error("Parent must be supported at the set of children")
  prob_mat = map(x->value.(probs(x)),children) #vec of vec of probabilities
  prob_vec = value.(probs(parent))
  DiscreteDist(supp,reduce(+,[prob_vec[i]*prob_mat[i] for i in 1:length(prob_vec)]))
end
"""
An amount of a currency, given by the
`Char`` type parameter, stored as an exact integer number of its
lowest denomination.
"""
struct Money{C}
  amount::Int
  Money{C}(n::N) where {C,N<:Number} = new{C}(Int(round(n)))
end
Money(c::Char,n::N) where {N<:Number} = Money{c}(Int(round(n)))
currency(m::Money{C}) where C = C
amount(m::Money) = m.amount
length(m::Money) = 1
###

###Boilerplate overloadings
for op in [:(+),:(-)]
  eval(:($op(a::Money{C},b::Money{D}) where {C,D} = 
    C == D ? Money(C,$op(amount(a),amount(b))) : 
    error("Currencies do not match!")))
end
for op in [:(+),:(-),:(*)]
  eval(:($op(p::Probability{P},q::Probability{Q}) where {P,Q} = Probability($op(value(p),value(q)))))
end
≈(p::Probability{P},q::Probability{Q}) where {P,Q} = value(p) ≈ value(q)

/(a::Money{C},b::Money{D}) where {C,D} = C == D ? amount(a)/amount(b) : error("Currencies do not match!")
*(a::Number,b::Money{C}) where C = Money(C,a*amount(b))
*(a::Money{C},b::Number) where C = Money(C,b*amount(a))
/(a::Money{C},b::Number) where C = Money(C,amount(a)/b)
/(a::Probability,b::Number) = Probability(value(a)/b)
*(a::Number,b::Probability) = Probability(a*value(b))
*(a::Probability,b::Number) = Probability(b*value(a))
*(a::Probability,b::Money) = Money(currency(b),value(a)*amount(b))
*(a::Money{C},b::Probability) where C = Money(C,value(b)*amount(a))
isless(a::Money{C},b::Money{D}) where {C,D} = 
  C == D ? amount(a) < amount(b) : error("Currencies do not match!")
###End boilerplate


###Data structures specific to the case
struct FundingRecipient{S}
  name::S
end
id(f::FundingRecipient) = f.name

"""
A single tranche of funding, given by a total size and a distribution
over a set of funding recipients.
"""
struct Tranche{C,S,P<:Real}
  size::Money{C}
  dist::DiscreteDist{FundingRecipient{S},P}
  Tranche(size::Money{C},dist::DiscreteDist{FundingRecipient{S},P}) where {C,S,P<:Real} =
  new{C,S,P}(size,dist)
end
size(t::Tranche) = t.size
dist(t::Tranche) = t.dist

"Priority-ordered list of giving tranches."
struct PreferenceList{C,S,P}
  tranches::Vector{Tranche{C,S,P}}
  #needs full interconvertibility with MVFs
  function PreferenceList(ts::Vector{Tranche{C,S,P}}) where {C,S,P} 
    new{C,S,P}(ts)
  end
end
tranches(plist::PreferenceList) = plist.tranches
sizes(plist::PreferenceList) = size.(tranches(plist))
length(plist::PreferenceList) = length(tranches(plist))
###


max_budget(ts::AbstractVector{Tranche{C,S,P}}) where {C,S,P} = mapreduce(t->size(t),+,ts,init=Money{C}(0))
max_budget(prefs::PreferenceList{C,S,P}) where {C,S,P} = max_budget(tranches(prefs))

"""
Proportionally pad a preference list to the given intended max budget.
"""
function pad(plist::PreferenceList,new_budget::Money)
  ratio = amount(new_budget)/amount(max_budget(plist)) #XX  could implement /(Money,Money)
  #Make sure the max budget comes out to exactly new_budget
  new_tranches = map(tranches(plist)) do t Tranche(ratio*size(t),dist(t)) end
  approx_budget = max_budget(new_tranches)
  prelim_top_tranche = new_tranches[1]
  new_tranches[1] = Tranche(size(prelim_top_tranche)+
                            new_budget-approx_budget,
                            dist(prelim_top_tranche))
  PreferenceList(new_tranches)
end
pad(plist::PreferenceList{C},new_budget_int::Int) where C = pad(plist,Money{C}(new_budget_int))
    
###Main functionality
"""
Use a parent `PreferenceList` to distribute funds across grandchildren according to 
child `PreferenceList`s.
"""


function distribute_down(parent_list::PreferenceList{C,T,P},child_lists::Vector{PreferenceList{C,S,P}}) where {C,T,P,S}
  new_tranches = Vector{Tranche{C,S,P}}()
  distribute_down(deepcopy(parent_list),deepcopy(child_lists),new_tranches,
    minimum([max_budget(parent_list);[max_budget(cl) for cl in child_lists]]))
  PreferenceList(new_tranches)
end
function distribute_down(parent_list::PreferenceList{C},child_lists,new_tranches,budget) where C
  @assert length(child_lists) > 0 && all(length(tranches(child_lists[i])) >0 for i in 1:length(child_lists))
  supp = support(dist(tranches(child_lists[1])[1]))
  while max_budget(new_tranches) < budget
    parent_tranche = tranches(parent_list)[1]
    parent_size = size(parent_tranche)
    child_tranches = map(clist->tranches(clist)[1],child_lists)
    child_sizes = map(size,child_tranches)
    child_dists = map(probs∘dist,child_tranches)
    #convex_combination takes 1/4 the time
    current_distribution = probs(convex_combination(dist(parent_tranche),map(dist,child_tranches),supp))
    #Calculate quantity it's safe to give out
    child_quantity = Money{C}(10^8)
    for (c,s) in [(c,s) for c in eachindex(supp), 
                            s in eachindex(child_lists)]
      cv = amount(child_sizes[s])*value(child_dists[s][c])/value(current_distribution[c])
      if 0 < cv <= 100000 child_quantity = Money{C}(100000); break end
      if cv > 0 child_quantity = min(child_quantity,Money{C}(cv)) end
     end
    current_quantity = min(parent_size,child_quantity)
    #println("current_quantity: $current_quantity")
    current_quantity > Money{C}(0) || break
    current_amounts = Money{C}.(amount(current_quantity) .* value.(current_distribution))
    flurrent_amounts = Float64.(amount.(current_amounts))
    push!(new_tranches,Tranche(current_quantity,DiscreteDist(supp,current_distribution)))
    #shrink parent and child tranches and modify current child distributions
    if current_quantity < parent_size
      tranches(parent_list)[1] = Tranche(parent_size-current_quantity,dist(parent_tranche))
    else
      parent_list = PreferenceList(tranches(parent_list)[2:end])
    end
    for i in eachindex(child_lists)
      clist = child_lists[i]
      if current_quantity < (s = child_sizes[i])
        new_targets = max.((amount(s)*value.(child_dists[i]).-flurrent_amounts),0)
        new_size = s-current_quantity#Money{C}(sum(new_targets))
        #Don't allow for probabilities below 10^3 in distributions
        new_dist = round.(1/sum(new_targets) * new_targets,digits=3)
        new_dist = Probability.(new_dist./sum(new_dist))
        tranches(clist)[1] = Tranche(s-current_quantity,DiscreteDist(supp,new_dist))
      else 
        child_lists[i] = PreferenceList(tranches(clist)[2:end])
      end
    end
  end
  new_tranches
end
end#module CalculateGiving
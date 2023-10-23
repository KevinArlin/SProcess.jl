module TestCalculateGiving
  using SProcess, Test, Random
  Random.seed!(13456534)
  seven_fifty = Money('$',750)
  six_oh_three = Money('$',603)
  ten_pounds_five_pence = Money('£',1005)

  @testset begin
  @test seven_fifty+six_oh_three == Money('$',1353)
  @test_throws ErrorException seven_fifty+ten_pounds_five_pence
  
  @test Probability(0.7)+Probability(0.2) ≈ Probability(0.9)
  @test_throws ErrorException Probability(1.1)
  @test_throws ErrorException Probability(0.8)+Probability(0.3)
  example_dist = DiscreteDist(FundingRecipient.(["Topos","AMF","MOMA"]),[1/6,1/3,1/2])
  @test_throws ErrorException DiscreteDist(["hey","there"],[1.2,0.1])
  recips = [FundingRecipient("Topos"),FundingRecipient("AMF"),FundingRecipient("MOMA")]
  t = Tranche(Money('$',100000),example_dist)
  p = PreferenceList([t,t])
  @test max_budget(p) == Money('$',200000)
  @test max_budget(pad(p,312356)) == Money('$',312356)
  
  function random_dist(supp)
    n = length(supp)
    unnormalized = rand(n)
    DiscreteDist(supp,unnormalized/sum(unnormalized))
  end

  children = [random_dist(recips) for _ in 1:100]
  par = random_dist(children)
  combo = convex_combination(par,children,recips)
  @test abs(value(probs(combo)[1]) - value(sum(probs(par).*[probs(c)[1] for c in children])))<10^-5

  function random_plist(support::Vector{FundingRecipient{S}},budget::Money{C},ntranches=10) where {S,C}
    tranche_sizes = Money{C}.(amount(budget).*(value.(probs(random_dist(collect(1:ntranches))))))
    tranche_dists = [random_dist(support) for _ in 1:ntranches]
    tranches = map(t->Tranche(t...),zip(tranche_sizes,tranche_dists))
    pad(PreferenceList(tranches),budget)
  end
  nRecips = 100
  nAssessors = 10
  budgetEach = 100000000
  recips = [FundingRecipient(i) for i in 1:nRecips]
  plist = random_plist(recips,Money{'$'}(budgetEach))
  @test sum(sizes(plist)) == Money{'$'}(budgetEach)
  child_lists = [random_plist(recips,Money('$',budgetEach),25) for _ in 1:nAssessors]
  parent_list = random_plist(map(FundingRecipient,collect(1:nAssessors)),Money('$',budgetEach),50)
  @time new_list = distribute_down(parent_list,child_lists)
  @test probs(dist(tranches(new_list)[1])) == probs(convex_combination(dist(tranches(parent_list)[1]),map(cl->dist(tranches(cl)[1]),child_lists),support(dist(tranches(child_lists[1])[1]))))

  end #test set
end #test module
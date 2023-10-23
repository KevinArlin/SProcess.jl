module SProcess
using Reexport
include("CalculateGiving.jl")
@reexport using .CalculateGiving
end # module SProcess

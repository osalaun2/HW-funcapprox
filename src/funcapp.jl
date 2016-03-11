

module funcapp


using FastGaussQuadrature
using PyPlot
using ApproXD
import ApproXD: getBasis, BSpline
using ApproxFun
using CompEcon













# use chebyshev to interpolate this:

		function q1(n)

			f(x) = x + 2*(x^2) - exp(-x) # function to be approximated
			deg = n-1
			a = -3.
			b = 3. # bounds for x

# Evaluating the Chebyshev Polynomial and Basis Function

			T(x,deg) = cos(acos(x)*deg) # definition of Tj(z)
			z(x,a,b) = 2 * (x - a)/(b - a) - 1	# normalizing x belonging to [a,b] to a basis defined in [-1,1]

			nodes = gausschebyshev(n)[1]*3
			mapped_nodes = Float64[(nodes[1][i]+1)*(b-a) / 2 + a for i=1:n]  # mapping nodes from [-1,1] to x belonging to [a,b]

#	Constructing Phi as T evaluated at the Chebyshev Nodes

			Phi = Float64[cos(((n-i+0.5)*(j-1)*pi)/n) for i in 1:n, j in 1:(deg+1)]
			y = f(mapped_nodes)
			c = inv(convert(Array{Float64,2},Phi))*y # getting the c vector

# Predict n_new=50 new points in [−3,3] using the interpolator

			n_new = 50
			new = linspace(a,b,n_new) # get 50 points
			Phi_new = Float64[T(z(new[i],a,b),j) for i in 1:n_new, j in 0:deg]


			y_new = Phi_new * c
			y_true = f(new)
			error = y_new - y_true # deviation between approximation and true function

# Plot

			figure()
			plot(1:n_new,error)
			title("Deviation in approximation from the true f")
			return error

# Automated test
			max = maximum(error)
			println("Deviation in approximation from the true f")

			if max < 1e-9
				println("The approximation error is smaller than 1e-9 and is equal to $max")
 			else
 				println("Error : The approximation error is larger than 1e-9 and is equal to $max")
 			end

		end














	function q2(n)

		S=Chebyshev([-3,3])
		x = points(S,n) # get the n points between -3 and 3 to extrapolate.
		s = x + 2*(x^2) - exp(-x)
		n_new = 50
		new_s = linspace(-3,3,n_new) # get 50 points
		AF = Fun(ApproxFun.transform(S,s),S)
		ApproxFun.plot(graph; title="Question 1 with ApproxFun package")


	end















	# plot the first 9 basis Chebyshev Polynomial Basis Functions
	function q3()


	x = linspace(-1,1,1000)
	che = [cos(acos(x)j) for j in 0:8] # We need the first nine Chebyshev basis functions
	n=1000
	m=9
	Phi = Float64[cos((n-i+0.5)*(j-1)*pi/n) for i=1:n,j=1:m]
	fig,axes = subplots(3,3,figsize=(10,5))

	for i in 1:3
		for j in 1:3
			ax = axes[j,i]
			count = i+(j-1)*3

			ax[:plot](x,che[i+(j-1)*3])
			ax[:set_title]("Basis function $(count-1)")
			ax[:xaxis][:set_visible](false)
			ax[:yaxis][:set_visible](false)
			ax[:set_xlim](-1.1,1.1)
			ax[:set_ylim](-1.1,1.1)

		end
	end

	fig[:canvas][:draw]()

	end












	ChebyT(x,deg) = cos(acos(x)*deg)
	unitmap(x,lb,ub) = 2.*(x.-lb)/(ub.-lb) - 1	#[a,b] -> [-1,1]

	type ChebyType
		f::Function # function to approximate
		nodes::Union{Vector,LinSpace} # evaluation points
		basis::Matrix # basis evaluated at nodes
		coefs::Vector # estimated coefficients

		deg::Int 	# degree of chebypolynomial
		lb::Float64 # bounds
		ub::Float64

		# constructor
		function ChebyType(_nodes::Union{Vector,LinSpace},_deg,_lb,_ub,_f::Function)
			n = length(_nodes)
			y = _f(_nodes)
			_basis = Float64[ChebyT(unitmap(_nodes[i],_lb,_ub),j) for i=1:n,j=0:_deg]
			_coefs = _basis \ y  # type `?\` to find out more about the backslash operator. depending the args given, it performs a different operation
			# create a ChebyType with those values
			new(_f,_nodes,_basis,_coefs,_deg,_lb,_ub)
		end
	end

	# function to predict points using info stored in ChebyType
	function predict(Ch::ChebyType,x_new)

		true_new = Ch.f(x_new)
		basis_new = Float64[ChebyT(unitmap(x_new[i],Ch.lb,Ch.ub),j) for i=1:length(x_new),j=0:Ch.deg]
		basis_nodes = Float64[ChebyT(unitmap(Ch.nodes[i],Ch.lb,Ch.ub),j) for i=1:length(Ch.nodes),j=0:Ch.deg]
		preds = basis_new * Ch.coefs
		preds_nodes = basis_nodes * Ch.coefs

		return Dict("x"=> x_new,"truth"=>true_new, "preds"=>preds, "preds_nodes" => preds_nodes)
	end













	function q4a(deg=(5,9,15),lb=-1.0,ub=1.0)

		f(x) = 1/(1+25*(x^2))
		h = linspace(-5,5,1000) # 1000 points to extrapolate

		lb,ub = (-5,5)

		fig,axes = subplots(1,2,figsize=(12,6)) # plot with 2 panels
		n = deg[j] + 1

		ax = axes[1,1]
 		ax[:set_title]("Chebyshev nodes")

		for j in 1:3


		nodes_5 = gausschebyshev(n)[1]*5
		nodes_9 = gausschebyshev(n)[1]*9
		nodes_15 = gausschebyshev(n)[1]*15

		x = predict(ChebyType(nodes,n,lb,ub,f),h)["x"]
		y_true = predict(ChebyType(nodes,n,lb,ub,f),h)["true function"]
		y_5 = predict(ChebyType(nodes_5,n,lb,ub,f),h)["approx_5"]
		y_9 = predict(ChebyType(nodes_9,n,lb,ub,f),h)["approx_9"]
		y_15 = predict(ChebyType(nodes_15,n,lb,ub,f),h)["approx_15"]

# Plot for each line
		ax[:plot](x,y_true)
		ax[:plot](x,y_5)
		ax[:plot](x,y_9)
		ax[:plot](x,y_15)
		ax[:set_ylim](-0.1,1.1)

	end

end








	function q4b()

		f(x) = 1/(1+25*(x^2))
		l = linspace(-5,5,1000) # 1000 points between -5 and 5
		true_f = f(l)

# Version 1: 13 equally spaced knots
		bs = BSpline(13,3,-5,5) # 13 knots, order 3 (cubic), lower and upper bounds

 		base_1 = full(getBasis(collect(linspace(-5,5,65)),bs)) # computation of the base
 		y = f(linspace(-5,5,65)) # 65 equidistant points
		coef_1 = base_1 \ y 							 # in function q1(), y = f(mapped_nodes)

 		# Simulate the  function
 		base_1 = full(getBasis(collect(l),bs)) # get the basis functions
 		apprx1_f = base_1 * coef_1
 		dev_1 = true_f - apprx1_f


# Version 2: knots concentrated toward 0

		knots2 = vcat(collect(linspace(-5,-1,3)), collect(linspace(-0.5,0.5,7)), collect(linspace(1,5,3)))
 		bs = BSpline(knots2,3)

		# Get the coefficients
		base_2 = full(getBasis(collect(linspace(-5,5,65)),bs)) # computation of the base
 		y = f(linspace(-5,5,65))
 		coef_2 = base_2 \ y

 		# Simulation of the function
 		base_2 = full(getBasis(collect(l),bs)) # get the basis functions
 		apprx2_f = base_2 * coef_2
 		dev_2 = true_f - apprx2_f

 # Plot
 		fig,axes = subplots(1,2,figsize=(10,5))

 		ax = axes[1,1]
 		ax[:set_title]("True function")
 		ax[:plot](l, true_f)

 		ax = axes[2,1]
 		ax[:set_title]("Deviation from true function")
 		ax[:plot](l, dev_1)
 		ax[:scatter](knots2,f_knots)
 		ax[:plot](l,dev_2)
 		fig[:canvas][:draw]()

 		println("Concentrated nodes seem to reduce the deviation")



	end











	function q5()

	# missing

	end











		# function to run all questions
	function runall()
		println("running all questions of HW-funcapprox:")
		q1()
		q2()
		q3()
		q4a()
		q4b()
		q5()
	end


end

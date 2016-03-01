	

module funcapp


	using FastGaussQuadrature # to get chebyshevnodes
	using PyPlot
	import ApproXD: getBasis, BSpline
	using Distributions
	using ApproxFun
	using CompEcon

	ChebyT(x,deg) = cos(acos(x)*deg)
	unitmap(x,lb,ub) = 2.*(x.-lb)/(ub.-lb) - 1	#[a,b] -> [-1,1]

	# use chebyshev to interpolate this:
	function q1(n)
		f(x) = x .+ 2x.^2 - exp(-x)
		deg = n-1
		lb,ub = (-3.0,3.0)
		z = gausschebyshev(n)
		xnodes = Float64[(z[1][i]+1)*(ub-lb) / 2 + lb for i=1:n]  # map z[-1,1] to x[a,b]
		y = f(xnodes)
		Phi = Float64[cos((n-i+0.5)*(j-1)*pi/n) for i=1:n,j=1:(deg+1)]
		c = Phi \ y  # solve for coefficients by solving interpolation equation. type `?\` to find out more
		n_new = 50
		xnew = linspace(lb,ub,n_new) # get 50 linspace points 
		# how to evaluate Phi at xnew?
		# evaulate Cheby basis at the new poitns
		Phi_xnew = Float64[ChebyT(unitmap(xnew[i],lb,ub),j) for i=1:n_new,j=0:deg]
		ynew = Phi_xnew * c
		ytrue = f(xnew)
		# find maximal error
		err = ynew - ytrue
		# plot
		figure()
		plot(1:n_new,err)
		title("Cheby error")
		return err
	end

	function q2(n)
		# use ApproxFun.jl to do the same:
		figure("simpleCheby with Approxfun")
		x = ApproxFun.Fun(ApproxFun.Interval(-3,3.0))
		lb,ub = (-3.0,3.0)
		g = x + 2x^2 - exp(-x)
		n_new = 50
		xnew = linspace(lb,ub,n_new) # get 50 linspace points 
		ApproxFun.plot(g)
	end


	# plot the first 9 basis Chebyshev Polynomial Basisi Fnctions
	function q3()

		n = 200
		m=9
		Phi = Float64[cos((n-i+0.5)*(j-1)*pi/n) for i=1:n,j=1:m]
		fig,axes = subplots(3,3,figsize=(10,5))

		for i in 1:3
			for j in 1:3
				ax = axes[j,i]
				count = i+(j-1)*3

				ax[:plot](Phi[:,count])
				ax[:set_title]("Basis $(count-1)")
				ax[:xaxis][:set_visible](false)
				ax[:set_ylim](-1.1,1.1)
				ax[:xaxis][:set_major_locator]=matplotlib[:ticker][:MultipleLocator](1)
				ax[:yaxis][:set_major_locator]=matplotlib[:ticker][:MultipleLocator](1)
			end
		end
		return fig
	end

	type ChebyType
		f::Function # fuction to approximate 
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
			# create a ChebyComparer with those values
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

		runge(x) = 1.0./(1+25.*x.^2)

		@assert length(deg)==3

		# set up the figure
		fig,axes   = subplots(1,2,figsize=(15,8),sharey=true)
		xnew = linspace(lb,ub,500)

		# compares what
		colors   = ["blue","red","green"]

		ax = axes[1]
		ax[:plot](xnew,runge(xnew),color="black",lw=2,label="runge")
		for j in 1:length(deg)
			nodes = linspace(lb,ub,deg[j]+1)
			C = ChebyType(nodes,deg[j],lb,ub,runge)
			p = predict(C,xnew)
			ax[:plot](xnew,p["preds"],label="deg=$(deg[j])",color=colors[j],lw=2)
		end
		ax[:legend](loc="upper center")	# add legend (taking labels)
		ax[:grid]()
		ax[:set_title]("Uniform Nodes")

		ax = axes[2]
		ax[:plot](xnew,runge(xnew),color="black",lw=2,label="runge")
		for j in 1:length(deg)
			n = deg[j] + 1
			z = gausschebyshev(n)
			nodes = Float64[(z[1][i]+1)*(ub-lb) / 2 + lb for i=1:n]
			C = ChebyType(nodes,deg[j],lb,ub,runge)
			p = predict(C,xnew)
			ax[:plot](xnew,p["preds"],label="deg=$(deg[j])",color=colors[j],lw=2)
		end
		ax[:legend](loc="upper center")	# add legend (taking labels)
		ax[:grid]()
		ax[:set_title]("Chebyshev Nodes")

	end
	

	# Do a high-dim approx with interploations.jl
	# function f(x,y,z,w) 
	# 	sin(sqrt(x^2+y^2)) + (z-w)^3
	# end
	# n = 7
	# space = collect(linspace(-1,1.0,n))
	# A = [f(x,y,z,w) for x in space,  y in space , z in space,  w in space]
	# itp = interpolate(A, (BSpline(Linear()), BSpline(Linear()),BSpline(Linear()),BSpline(Linear())), OnGrid())




	function q4b()

		# compare 2 knot vectors with runge's function
		ub,lb = (5,-5)
		runge(x) = 1.0./(1+25.*x.^2)

		nknots = 13
		deg = 3
		bs1 = BSpline(nknots,deg,lb,ub)	# equally spaced knots in [lb,ub]
		nevals = 5 * bs1.numKnots # get nBasis < nEvalpoints

		# scaled knots
		G(k,s) = GeneralizedPareto(k,s,0)
		pf(k,s) = quantile(GeneralizedPareto(k,s,0),linspace(0.05,cdf(G(0.5,1),5),6))
		myknots = vcat(-reverse(pf(0.5,1)),0.0,pf(0.5,1))

		bs2 = BSpline(myknots,deg)

		# get coefficients for each case

		eval_points = collect(linspace(lb,ub,nevals))  
		c1 = getBasis(eval_points,bs1) \ runge(eval_points)
		c2 = getBasis(eval_points,bs2) \ runge(eval_points)	# now \ implements the penrose-moore invers. see `pinv`. regression.

		# look at errors over entire interval
		test_points = collect(linspace(lb,ub,1000));
		truth = runge(test_points);
		e1 = getBasis(test_points,bs1) * c1 - truth;
		e2 = getBasis(test_points,bs2) * c2 - truth;
		figure(figsize=(8,7))
		subplot(211)
		plot(test_points,truth,lw=2)
		ylim(-0.2,1.2)
		grid()
		title("Runge's function")
		subplot(212)
		plot(test_points,e1,label="equidistant",color="blue")
		plot(test_points,e2,label="concentrated",color="red")
		plot(unique(bs1.knots),zeros(nknots),color="blue","+")
		plot(myknots,ones(nknots)*ylim()[1]/2,color="red","o")
		ylim(minimum(e1)-0.1,maximum(e1)+0.1)
		grid()
		legend(loc="upper right",prop=Dict("size"=>8))
		title("Errors in Runge's function")

	end

	function q5()

		f(x) = abs(x).^0.5
		lb,ub = (-1.0,1.0)
		nknots = 13
		deg = 3
	    params1 = SplineParams(linspace(lb,ub,nknots),0,deg)  # 0: no derivative
		nevals = 5 * length(params1.breaks) # get nBasis < nEvalpoints

		# myknots
		myknots = vcat(linspace(-1,-0.1,5),0,0,0,	linspace(0.1,1,5))
	    params2 = SplineParams(myknots,0,deg)  # 0: no derivative

		# get coefficients for each case
		eval_points = collect(linspace(lb,ub,nevals))  
		c1 = CompEcon.evalbase(params1,eval_points)[1] \ f(eval_points)
		c2 = CompEcon.evalbase(params2,eval_points)[1] \ f(eval_points)

		# look at errors over entire interval
		test_points = collect(linspace(lb,ub,1000));
		truth = f(test_points);
		p1 = CompEcon.evalbase(params1,test_points)[1] * c1;
		p2 = CompEcon.evalbase(params2,test_points)[1] * c2;
		e1 = p1 - truth;
		e2 = p2 - truth;

		fix,axes = subplots(1,3,figsize=(13,5))
		ax = axes[1]
		ax[:plot](test_points,truth,lw=2,color="black")
		ax[:set_title]("truth")
		ax[:set_ylim](0,1.2)
		ax[:grid]()

		ax = axes[2]
		ax[:grid]()
		ax[:plot](test_points,p1,label="equidistant",color="blue")
		ax[:plot](test_points,p2,label="stacked",color="red")
		ax[:legend](loc="upper center")
		ax[:set_ylim](0,1.2)
		ax[:set_title]("Approximation")

		ax = axes[3]
		ax[:grid]()
		ax[:plot](test_points,e1,label="equidistant",color="blue")
		ax[:plot](test_points,e2,label="stacked",color="red")
		ax[:set_title]("errors")
	end


		# function to run all questions
	function runall()
		println("running all questions of HW-funcapprox:")
		q1(15)
		q2(15)
		q3()
		q4a()
		q4b()
		q5()
	end


end


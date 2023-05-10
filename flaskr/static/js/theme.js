$(document).ready(function() {

	// preloader 
	if($('.preloader').length) {
		$('.preloader').delay(750).fadeOut(500)
	}

	if ($('.main-menu-list').length) {
		$('.main-menu-list').singlePageNav({
			offset: $('.main-menu').outerHeight(),
			threshold: 120,
			speed: 400,
			currentClass: 'active',
			easing: 'swing',
			filter: ':not(.external-page)',
		});
	}

	//scroll to target

	if ($(".scroll-to-target").length) {
		var progressPath = document.querySelector('.scroll-to-top path');
		var pathLength = progressPath.getTotalLength();
		progressPath.style.transition = progressPath.style.WebkitTransition = 'none';
		progressPath.style.strokeDasharray = pathLength + ' ' + pathLength;
		progressPath.style.strokeDashoffset = pathLength;
		progressPath.getBoundingClientRect();
		progressPath.style.transition = progressPath.style.WebkitTransition = 'stroke-dashoffset 0 linear';		
		var updateProgress = function () {
			var scroll = $(window).scrollTop();
			var height = $(document).height() - $(window).height();
			var progress = pathLength - (scroll * pathLength / height);
			progressPath.style.strokeDashoffset = progress;
		}
		updateProgress();
		$(window).scroll(updateProgress);

		$(".scroll-to-target").on("click", function () {
		var target = $(this).attr("data-target");
		// animate
		$("html, body").animate({
			scrollTop: $(target).offset().top,
			},1000);
		return false;
	  });
	}

	//  mobile nav toggler
	if ($('.mobile-menu-toggler').length) {
		$('.mobile-menu-toggler').on('click',function() {
			$(this).toggleClass('active');
			if (!$('.main-menu-list').is(':visible')) {
				$('.main-menu-list').slideDown(500)
			} else {
				$('.main-menu-list').slideUp(500)
				setTimeout(function() {
					$('.main-menu-list').removeAttr('style')
				},600)
			}
			$('.main-menu-list').addClass('active')
		})
	}


	if ($('.main-menu-list').length) {
		$('.main-menu-list li a').on('click',function() {
			 $('.main-menu-list').slideUp(500);
			 $('.mobile-menu-toggler').toggleClass('active');
		});		 
	}	


	//Slider One
	if ($('.slider-one-swiper').length) {
		$('.slider-one-swiper').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			animateOut: 'animate__slideOutDown',
			animateIn: 'animate__fadeIn',
			nav:false,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
		})
	}

	//Slider Two
	if ($('.slider-two-swiper').length) {
		$('.slider-two-swiper').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			animateOut: 'animate__slideOutDown',
			animateIn: 'animate__fadeIn',
			nav:false,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
		})
	}
	//Slider Three
	if ($('.slider-three-swiper').length) {
	$('.slider-three-swiper').owlCarousel({
		loop:true,
		margin:0,
		items:1,
		animateOut: 'animate__slideOutDown',
		animateIn: 'animate__fadeIn',
		nav:false,
		dots:false,
		autoplay:true,
		autoplayTimeout: 7000,
	})
	}
	//Slider Four
	if ($('.slider-four-swiper').length) {
		$('.slider-four-swiper').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			animateOut: 'animate__slideOutDown',
			animateIn: 'animate__fadeIn',
			nav:false,
			autoHeight:true,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
		})
	}
	//Slider Five
	if ($('.slider-five-swiper').length) {
		$('.slider-five-swiper').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			animateOut: 'animate__slideOutDown',
			animateIn: 'animate__fadeIn',
			nav:false,
			autoHeight:true,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
		})
	}
	//Slider Six
	if ($('.slider-six-swiper').length) {
		var sldierSix = $('.slider-six-swiper')
		sldierSix.owlCarousel({
			loop:true,
			margin:0,
			items:1,
			animateOut: 'animate__slideOutDown',
			animateIn: 'animate__fadeIn',
			nav:false,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
		})
		sldierSix.on('change.owl.carousel', function(e) {
			setTimeout(slideCouner,1500)
			$(".counter").html(0)
		});
		setTimeout(slideCouner,1500)
	}
	function slideCouner() {
		if ($(".counter").length) {
			$(".counter").appear(function() {
			var countTo = $(this).attr('data-count')
			$(this).counter({
				autoStart: true,
				duration: 1000,
				countFrom: 0,
				countTo: countTo,
				runOnce: false,
				easing: false,
			});
			$(this).counter("start",{easing:false})
			});
		}
	}

	if ($('.client-one-carousel').length) {
		$('.client-one-carousel').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			nav:false,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
			responsive: {
				0: {
					margin: 30,
					items: 2
				},
				375: {
					margin: 30,
					items: 2
				},
				991: {
					margin: 40,
					items: 3
				},
				1199: {
					margin: 80,
					items: 4
				}
			}
		})
	}

	// Client Two carousel
	if ($('.client-two-carousel').length) {
		$('.client-two-carousel').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			nav:false,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
			responsive: {
				0: {
					margin: 30,
					items: 1
				},
				375: {
					margin: 30,
					items: 2
				},
				575: {
					margin: 30,
					items: 3
				},
				991: {
					margin: 40,
					items: 4
				},
				1199: {
					margin: 80,
					items: 5
				}
			}
		})
	}

	// Team carousel
	if ($('.team-three-carousel').length) {
		$('.team-three-carousel').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			center:true,
			nav:true,
			navText: ['<i class="fas fa-chevron-left"></i>','<i class="fas fa-chevron-right"></i>'],
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
				responsive: {
				575: {
					margin: 30,
					items: 1
				},
				767: {
					margin: 30,
					items: 2,
					center: false
				},
				1199: {
					margin: 55,
					items: 3
				}
			}
		})
	}
	// Team carousel
	if ($('.nft-one-carousel').length) {
		$('.nft-one-carousel').owlCarousel({
			loop:true,
			margin:30,
			items:1,
			center:true,
			nav:false,
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
			responsive: {
				575: {
					margin: 30,
					items: 1
				},
				767: {
					margin: 30,
					items: 2,
					center: false
				},
				1199: {
					margin: 30,
					items: 3
				}
			}
		})
	}
	
	//Testimonial One
	if ($('.testimonial-one-carousel').length) {
		$('.testimonial-one-carousel').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			nav:true,
			navText: ['<i class="fas fa-chevron-left"></i>','<i class="fas fa-chevron-right"></i>'],
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
		})
	}

	// Testimonial Three
	if ($('.testimonial-three-carousel').length) {
		$('.testimonial-three-carousel').owlCarousel({
			loop:true,
			margin:0,
			items:1,
			center:true,
			nav:true,
			navText: ['<i class="fas fa-chevron-left"></i>','<i class="fas fa-chevron-right"></i>'],
			dots:false,
			autoplay:true,
			autoplayTimeout: 7000,
			responsive: {
				575: {
					margin: 30,
					items: 1
				},
				767: {
					margin: 30,
					items: 2,
					center:false
				},
				1199: {
					margin: 45,
					items: 3
				}
			}
		});
	}

	if ($('.testimonial-four-swiper').length) {
		$('.testimonial-four-thumbnail-swiper').slick({
			dots: false,
			prevArrow: false,
			nextArrow: false,
			vertical: true,
			focusOnSelect: true,
			asNavFor: '.testimonial-four-swiper',
			slidesToShow: 3,
			slidesToScroll: 1,
			infinite:false,
		});
		$('.testimonial-four-swiper').slick({
			dots: false,
			prevArrow: false,
			infinite:false,
			nextArrow: false,
			autoplay: true,
			autoplaySpeed: 2000,
			asNavFor: '.testimonial-four-thumbnail-swiper',
			slidesToShow: 1,
		});

		// Remove active class from all thumbnail slides
		 $('.testimonial-four-thumbnail-swiper .slick-slide').removeClass('slick-active');

		 // Set active class to first thumbnail slides
		 $('.testimonial-four-thumbnail-swiper .slick-slide').eq(0).addClass('slick-active');

		 // On before slide change match active thumbnail to current slide
		 $('.testimonial-four-swiper').on('beforeChange', function (event, slick, currentSlide, nextSlide) {
			var mySlideNumber = nextSlide;
			$('.testimonial-four-thumbnail-swiper .slick-slide').removeClass('slick-active');
			$('.testimonial-four-thumbnail-swiper .slick-slide').eq(mySlideNumber).addClass('slick-active');
		});
	}

	/* accordian */

	if ($('.accordian-group').length) {
	$('.accordian-item:not(.active) .accordian-content').hide()
	$('.accordian-title').on('click', function() {
		$(this).parents('.accordian-group').find('.accordian-item').removeClass('active')
		$(this).parents('.accordian-item').addClass('active')
		$(this).parents('.accordian-group').find  ('.accordian-content').stop().slideUp(500)
		$('.accordian-item.active .accordian-content').stop().slideDown(500)
	})
	} 

	//chart
	if ($('#token-one-chart').length) {
	var chartData = {
		labels:[],
		data: [],
		color: []
	};

	$('.token-one-list-item').each(function() {
		var singleLabel = $(this).attr('data-token-label')
		var singleData = $(this).attr('data-token-percentage')
		var singleColor = $(this).attr('data-token-color')
		chartData.labels.push(singleLabel)
		chartData.data.push(singleData)
		chartData.color.push(singleColor)
	})
	const ctx = $('#token-one-chart')
	if (chartData.data.length != 0) {
		$('.token-one-chart-wrapper').addClass('chart-loaded')
		const tokenChart = new Chart(ctx, {
			type: 'doughnut',
			data: {
				labels: chartData.labels,
				datasets: [{
				label: ' ',
				data: chartData.data,
				backgroundColor: chartData.color,
				borderWidth: 0,
				cutout: '47%'
				}]
			},
			options: {
				plugins: {
				legend: {
					display: false
				},
				}
			},
		});
	}
	} // /.token.one

	//token two chart
	if ($('#token-two-chart').length) {
		var chartData = {
			labels:[],
			data: [],
			color: []
		};

		$('.token-two-list-item').each(function() {
			var singleLabel = $(this).attr('data-token-label')
			var singleData = $(this).attr('data-token-percentage')
			var singleColor = $(this).attr('data-token-color')
			chartData.labels.push(singleLabel)
			chartData.data.push(singleData)
			chartData.color.push(singleColor)
		})
		const ctx = $('#token-two-chart')
		if (chartData.data.length != 0) {
			$('.token-two-chart-wrapper').addClass('chart-loaded')
			const tokenChart = new Chart(ctx, {
				type: 'doughnut',
				data: {
					labels: chartData.labels,
					datasets: [{
					label: ' ',
					data: chartData.data,
					backgroundColor: chartData.color,
					borderWidth: 0,
					cutout: '47%'
					}]
				},
				options: {
					plugins: {
					legend: {
						display: false
					},
					}
				},
			});
		}
	}

	if ($('.ico-state-chart').length) {
		const c2 = document.getElementById('ico-state-chart');
		const ctx2 = c2.getContext('2d')
		// gradient 1
		var gradient = ctx2.createLinearGradient(0, 0, 0, 300);
		gradient.addColorStop(0.25, 'rgba(165,75,255,1)');
		gradient.addColorStop(0.95, 'rgba(165,75,255,0)');
		// gradient 2
		var gradient2 = ctx2.createLinearGradient(0, 0, 0, 250);
		gradient2.addColorStop(0, 'rgba(255,74,139,1)');
		gradient2.addColorStop(0.9, 'rgba(255,74,139,0');

		var axixGradient = ctx2.createLinearGradient(0, 0, 0, 230);
		axixGradient.addColorStop(0, 'rgba(40,38,77,1)');
		axixGradient.addColorStop(1, 'rgba(40,38,77,0)');

		const ICOChart = new Chart(ctx2, {
			type: 'line',
			data: {
				labels: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
				datasets: [{
					label: '',
					data: [0, 30, 45, 45, 35, 25, 35, 50, 70, 80, 70, 0],
					fill:true,
					backgroundColor: gradient,
					borderColor: '#A54BFF',
					lineTension: 0.35,
					pointRadius: 0,
				},
				{
					label: '',
					data: [20, 80, 100, 95, 80, 60, 50, 50, 60, 60, 50, 20],
					fill:true,
					backgroundColor: gradient2,
					borderColor: '#FF3E88',
					lineTension: 0.4,
				},]
			},
			options: {
				elements:{
					point:{
					radius: 0,
					}
				},
				plugins: {
					legend: {
					display: false
					},
				},
				scales: {
					x: {
						grid: {
							color: axixGradient,
							lineWidth: 6
						},
						border: {
							display: false
						},
						ticks: {
							display: false
						}
					},
					y: {
						display: false
					}
				}
			}
		});
	}

	if ($('.crypto-chart-inner').length) {
	// rendering chart in hidden tabs
		$('.crypto-tabs .tab-pane').css('display','block')
		$('.crypto-tabs .tab-pane').css('opacity','1')
		setTimeout(function() {
			$('.crypto-tabs .tab-pane').removeAttr('style')
		},20);

		$('.crypto-chart-inner').each(function(index) {
			const c2 = document.getElementsByClassName('crypto-chart-inner')[index];
			const ctx2 = c2.getContext('2d')
			if($(this).parent().siblings('.crypto-change').hasClass('profit')) {
			var gradient = ctx2.createLinearGradient(0, 0, 0, 50);
			gradient.addColorStop(0, 'rgba(148,238,168,1)');
			gradient.addColorStop(1, 'rgba(148,238,168,0)');
			var borderColor = '#64AB74';
			} else {
			var gradient = ctx2.createLinearGradient(0, 0, 0, 50);
			gradient.addColorStop(0, 'rgba(255,132,132,1)');
			gradient.addColorStop(1, 'rgba(255,132,132,0)');
			var borderColor = '#D23737';
			}

			const cryptoChart = new Chart(ctx2, {
				type: 'line',
				data: {
					labels: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
					datasets: [{
					label: '',
					data: [0, 30, 45, 45, 35, 25, 35, 50, 70, 80, 70, 0],
					fill:true,
						backgroundColor: gradient,
						borderColor: borderColor,
						lineTension: 0.35,
						borderWidth: 1,
						pointRadius: 0,
					}]
				},
				options: {
					elements:{
						point:{
							radius: 0,
						}
					},
					plugins: {
						legend: {
							display: false
						},
					},
					scales: {
						x: {
							display: false
						},
						y: {
							display: false
						}
					}
				}
			});
		})
	}

	if ($('#coin-chart').length) {
		$('#coin-chart').each(function(index) {
			const c2 = document.getElementById('coin-chart');
			const ctx2 = c2.getContext('2d')
			
			var gradient = ctx2.createLinearGradient(0, 0, 0, 140);
			gradient.addColorStop(0, 'rgba(255,255,255,1)');
			gradient.addColorStop(1, 'rgba(255,255,255,0)');
			var axixGradient = ctx2.createLinearGradient(0, 0, 0, 140);
			axixGradient.addColorStop(0, 'rgba(255,197,122,1)');
			axixGradient.addColorStop(1, 'rgba(255,255,255,0)');
			var borderColor = '#091B2C';
			

			const cryptoChart = new Chart(ctx2, {
				type: 'line',
				data: {
					labels: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
					datasets: [{
					label: '',
					data: [0, 15, 23, 23, 18, 13, 16, 25, 35, 40, 35, 0],
					fill:true,
					backgroundColor: gradient,
					borderColor: borderColor,
					lineTension: 0.35,
					borderWidth: 2,
					pointRadius: 0,
					pointHoverRadius: 13,
					pointBackgroundColor: borderColor
					}]
				},
				options: {
					elements:{
						point:{
							radius: 0,
						}
					},
					plugins: {
						legend: {
							display: false
						},
					},
					scales: {
						y: {
							grid: {
								color: axixGradient,
								lineWidth: 1
							},
							border: {
								display: false
							},
							ticks: {
								display: false,
								max: 50,
                stepSize: 10
							}
						},
						x: {
							display: false
						}
					}
				}
			});
		})
	}

	if ($('.coin-card-chart-inner').length) {
		$('.coin-card-chart-inner').each(function(index) {
			const c2 = document.getElementsByClassName('coin-card-chart-inner')[index];
			const ctx2 = c2.getContext('2d')
			if($(this).parent().siblings('.coin-card-difference').hasClass('profit')) {
			var gradient = ctx2.createLinearGradient(0, 0, 0, 60);
			gradient.addColorStop(0, 'rgba(148,238,168,1)');
			gradient.addColorStop(1, 'rgba(148,238,168,0)');
			var borderColor = '#46F56D';
			} else {
			var gradient = ctx2.createLinearGradient(0, 0, 0, 60);
			gradient.addColorStop(0, 'rgba(255,132,132,1)');
			gradient.addColorStop(1, 'rgba(255,132,132,0)');
			var borderColor = '#FF3838';
			}

			const cryptoChart = new Chart(ctx2, {
				type: 'line',
				data: {
					labels: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
					datasets: [{
					label: '',
					data: [20, 50, 45, 30, 35, 55, 35, 20, 20, 50, 70, 0],
					fill:true,
					backgroundColor: gradient,
					borderColor: borderColor,
					lineTension: 0.35,
					borderWidth: 3,
					pointRadius: 0,
					}]
				},
				options: {
					elements:{
						point:{
							radius: 0,
						}
					},
					plugins: {
						legend: {
							display: false
						},
					},
					scales: {
						x: {
							display: false
						},
						y: {
							display: false
						}
					}
				}
			});
		})
	}
	
	// count box
	if ($(".count-box").length) {
		$(".count-box").appear(
		function () {
			var $t = $(this),
			n = $t.find(".count-text").attr("data-stop"),
			r = parseInt($t.find(".count-text").attr("data-speed"), 10);
			if (!$t.hasClass("counted")) {
				$t.addClass("counted");
				$({
					countNum: $t.find(".count-text").text()
				}).animate({
					countNum: n
				}, {
					duration: r,
					easing: "linear",
					step: function () {
					$t.find(".count-text").text(Math.floor(this.countNum));
					},
					complete: function () {
					$t.find(".count-text").text(this.countNum);
					}
				});
			}
		}, {
			accY: 0
		});
	}

	// map dots
	if ($('.map-one-image').length) {
		$('.map-one-dot').on('click', function() {
		$('.map-one-dot').parent().removeClass('active')
		$(this).parent().addClass('active')
		$('.map-one-marker').removeClass('active')
		var mapId = $(this).parent().attr('data-map-id');
		$('.map-one-marker').each(function() {
			var mapTarget = $(this).attr('data-map-target')
			if (mapTarget === mapId) {
			$(this).addClass('active')
			}
		})
		})

		$('.map-one-marker').on('click', function() {
		$('.map-one-marker').removeClass('active')
		$(this).addClass('active')
		$('.map-one-dot').parent().removeClass('active')
		var mapTarget = $(this).attr('data-map-target');
		$('.map-one-dot-list-item').each(function() {
			var mapId = $(this).attr('data-map-id')
			console.log(mapTarget)
			if (mapTarget === mapId) {
			$(this).addClass('active')
			}
		})
		})
	}

	// countdown
	function countDown() {
		var endTime = $('.token-countdown').attr('data-date');      
		endTime = (Date.parse(endTime) / 1000);

		var now = new Date();
		now = (Date.parse(now) / 1000);

		var timeLeft = endTime - now;

		var days = String(Math.floor(timeLeft / 86400)).padStart(2,'0');
		var hours = String(Math.floor((timeLeft - (days * 86400)) / 3600)).padStart(2,'0');
		var minutes = String(Math.floor((timeLeft - (days * 86400) - (hours * 3600 )) / 60)).padStart(2,'0');
		var seconds = String(Math.floor((timeLeft - (days * 86400) - (hours * 3600) - (minutes * 60)))).padStart(2,'0');

		$(".token-countdown-item-number.day").html(days);
		$(".token-countdown-item-number.hour").html(hours);
		$(".token-countdown-item-number.min").html(minutes);
		$(".token-countdown-item-number.sec").html(seconds);    
		}

		setInterval(function() { countDown(); }, 0);
		
})


$(window).on('scroll',function() {
	if ($('.thm-sticky-header').length) {
		var headerScrollPos = 150;
		var sticky = $(".thm-sticky-header");
		if ($(window).scrollTop() > headerScrollPos) {
			sticky.addClass("thm-sticky");
			setTimeout(function() {
				sticky.addClass("sticky-fixed");
			},20)
		} else if ($(this).scrollTop() <= headerScrollPos) {
			sticky.removeClass("thm-sticky");
			// setTimeout(function() {
				sticky.removeClass("sticky-fixed");
			// },20)
		}
	}

	if ($(".scroll-to-top").length) {
		var strickyScrollPos = 100;
		if ($(window).scrollTop() > strickyScrollPos) {
			$(".scroll-to-top").addClass('active');
		} else if ($(this).scrollTop() <= strickyScrollPos) {
			$(".scroll-to-top").removeClass('active');
		}
	}
})
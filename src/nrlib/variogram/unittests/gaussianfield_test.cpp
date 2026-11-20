/// Unit tests for gaussian field simulation

#include <nrlib/grid/grid2d.hpp>
#include <nrlib/random/random.hpp>
#include <nrlib/variogram/gaussianfield.hpp>
#include <nrlib/variogram/variogram.hpp>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <memory>

using namespace NRLib;

BOOST_AUTO_TEST_SUITE( TestGaussianField )

BOOST_AUTO_TEST_CASE( AxisAlignedExtents )
{
  const double pi = std::acos(-1.0);
  std::unique_ptr<Variogram> v(Variogram::Create(Variogram::EXPONENTIAL, 1.5, 10.0, 2.0, 1.0));
  double x, y, z;
  v->GetAxisAlignedExtents(x, y, z);
  BOOST_CHECK_CLOSE(x, 10.0, 1e-10);
  BOOST_CHECK_CLOSE(y, 2.0, 1e-10);
  BOOST_CHECK_CLOSE(z, 1.0, 1e-10);

  v->SetAngles(pi / 4.0);
  v->GetAxisAlignedExtents(x);
  BOOST_CHECK_EQUAL(x, 10.0);
  v->GetAxisAlignedExtents(x, y);
  BOOST_CHECK_EQUAL(x, 10.0);
  BOOST_CHECK_EQUAL(y, 10.0);

  v->SetRanges(100.0, 20.0, 10.0);
  v->SetAngles(0.0, pi / 4.0);
  v->GetAxisAlignedExtents(x, y);
  BOOST_CHECK_EQUAL(x, 100.0);
  BOOST_CHECK_CLOSE(y, 20.0, 1e-10);
  v->GetAxisAlignedExtents(x, y, z);
  BOOST_CHECK_EQUAL(x, 100.0);
  BOOST_CHECK_CLOSE(y, 20.0, 1e-10);
  BOOST_CHECK_CLOSE(z, std::sqrt(5050.0), 1e-10);
  BOOST_CHECK_CLOSE(v->GetCorr(z * 99.0 / 101.0, 0.0, z), std::exp(-3.0), 1e-10);
}

BOOST_AUTO_TEST_CASE( AxisAlignedExtentsWithAzimuthAndDip )
{
  const double pi = std::acos(-1.0);
  const double azimuth = pi / 6.0;
  const double dip = pi / 9.0;
  const double c = std::cos(azimuth), s = std::sin(azimuth);
  const double cd = std::cos(dip), sd = std::sin(dip);
  std::unique_ptr<Variogram> v(Variogram::Create(
    Variogram::EXPONENTIAL, 1.5, 100.0, 20.0, 10.0, azimuth, dip));
  double x, y, z;
  v->GetAxisAlignedExtents(x, y, z);
  BOOST_CHECK_EQUAL(x, 100.0);
  BOOST_CHECK_EQUAL(y, 100.0);
  BOOST_CHECK_CLOSE(z, std::sqrt(10000.0 * sd * sd + 100.0 * cd * cd), 1e-10);
  const double tangent_horizontal = cd * sd * (10000.0 - 100.0) / z;
  BOOST_CHECK_CLOSE(v->GetCorr(c * tangent_horizontal, s * tangent_horizontal, z), std::exp(-3.0), 1e-9);
}

BOOST_AUTO_TEST_CASE( Sim2dComparison )
{
  Variogram * v = Variogram::Create(Variogram::EXPONENTIAL, 1.5, 1000.0, 500.0, 250.0);
  Grid2D<double> field;
  std::vector<Grid2D<double> > fields;
  Random::Initialize(123L);
  Simulate2DGaussianField(*v, 100, 20, 100, 20, field);
  unsigned long seed = Random::GetStartSeed();
  RandomGenerator * rg = new RandomGenerator(seed);
  Simulate2DGaussianField(*v, 100, 20, 100, 20, 1, fields, rg);
  delete rg;
  delete v;

  BOOST_TEST(field.GetStorage().size() == fields[0].GetStorage().size());
  // Check enough values to perform a valid comparison, but not
  // so many that it is computationally demanding.
  for (size_t i = 0; i < field.GetStorage().size(); i+=73) {
    BOOST_CHECK_CLOSE(field.GetStorage()[i], fields[0].GetStorage()[i], 1e-6);
  }
}

BOOST_AUTO_TEST_CASE( Sim2dValues )
{
  Variogram * v = Variogram::Create(Variogram::EXPONENTIAL, 1.5, 1000.0, 500.0, 250.0);
  std::vector<Grid2D<double> > fields;
  RandomGenerator * rg = new RandomGenerator(321);
  Simulate2DGaussianField(*v, 757, 20, 757, 20, 1, fields, rg);
  delete rg;
  delete v;
  const std::vector<double> & values = fields[0].GetStorage();
  // Values retrieved on a Win7 64-bit machine
  // Any lower tolerance may cause the closeness check
  // to fail.
  BOOST_CHECK_CLOSE(values[1] , -0.98538252281984096, 1e-11);
  BOOST_CHECK_CLOSE(values[11],  0.51097121453608729, 1e-11);
  BOOST_CHECK_CLOSE(values[33], -0.8397865987946399 , 1e-11);
  BOOST_CHECK_CLOSE(values[77],  0.78852851345913655, 1e-11);
}

BOOST_AUTO_TEST_CASE(Sim2dValuesSmall, * boost::unit_test::disabled())
{
  // Grid parameters and variogram range chosen such that
  // the FFT grid is larger than the simulation grid.
  // See the FFTGrid2D constructor and FindNewWithPadding
  // with the even argument set to true.
  Variogram * v = Variogram::Create(Variogram::EXPONENTIAL, 1.5, 150.0, 150.0, 100.0);
  std::vector<Grid2D<double> > fields;
  RandomGenerator * rg = new RandomGenerator(321);
  Simulate2DGaussianField(*v, 22, 50.0, 22, 50.0, 1, fields, rg);
  delete rg;
  delete v;
  const std::vector<double> & values = fields[0].GetStorage();

  BOOST_CHECK_CLOSE(values[0],           0.35718004270126918, 1e-11);
  BOOST_CHECK_CLOSE(values[22 * 6],      1.4074577493723737,  1e-11);
  BOOST_CHECK_CLOSE(values[22 * 12],     0.90197503251845967, 1e-11);
  BOOST_CHECK_CLOSE(values[22 * 18],     0.61475867165380083, 1e-11);
  BOOST_CHECK_CLOSE(values[22 * 22 - 1], 0.84887146362306831, 1e-11);
}

BOOST_AUTO_TEST_SUITE_END()

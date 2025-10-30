#ifndef INCLUDE_MATERIALS_VOL_MATERIALS_HPP_
#define INCLUDE_MATERIALS_VOL_MATERIALS_HPP_

#include "iso_vol_hyperelastic_material.hpp"

namespace commet_solve
{

template <typename Number = double>
class SquareAndLog : public VolHyperelasticMaterial<Number>
{
  public:
	SquareAndLog(const Number &k)
		: k(k) {};
	SquareAndLog(SquareAndLog &&) = delete;
	SquareAndLog(const SquareAndLog &) = delete;
	SquareAndLog &operator=(SquareAndLog &&) = delete;
	SquareAndLog &operator=(const SquareAndLog &) = delete;
	~SquareAndLog() = default;

	const Number k;

	void evaluate_vol(const Number &J, Number &strain_energy, Number &denergy_dJ, Number &d2energy_dJ2) const final
	{
		strain_energy += k * (J * J - 1 - 2 * log(J)) / 4;
		denergy_dJ = k * (J - 1 / J) / 2;
		d2energy_dJ2 = k * (1 + 1 / (J * J)) / 2;
	};

  private:
};

template <typename Number = double>
class Square : public VolHyperelasticMaterial<Number>
{
  public:
	Square(const Number &k)
		: k(k) {};
	Square(Square &&) = delete;
	Square(const Square &) = delete;
	Square &operator=(Square &&) = delete;
	Square &operator=(const Square &) = delete;
	~Square() = default;

	const Number k;

	void evaluate_vol(const Number &J, Number &strain_energy, Number &denergy_dJ, Number &d2energy_dJ2) const final
	{

		const Number J_1 = J - 1;
		strain_energy += k * J_1 * J_1 / 2.;
		denergy_dJ = k * J_1;
		d2energy_dJ2 = k;
	};

  private:
};

} // namespace commet_solve

#endif // INCLUDE_MATERIALS_VOL_MATERIALS_HPP_
